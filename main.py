"""Play loop.

    python main.py --learn        # play, and ask you whenever a card can't be read with confidence
    python main.py                # autonomous play
    python main.py --dry-run      # read the screen once and print the decision, never tap
    python main.py --debug        # also save annotated screenshots to debug/
    python main.py --levels 25    # stop after 25 wins

Stop with Ctrl+C. Results are appended to runs.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from bot import geometry as G
from bot.adb import Phone
from bot.brain import WILD, Free, LevelMemory, first_failed_tap, plan
from bot.debug import annotate
from bot.learn import ask_rank
from bot.screens import ScreenBook
from bot.vision import NAMES, GlyphBook, detect_scene, image_diff, pile_matches, read_game, wild_card


def save(folder: str, img, name: str) -> str:
    Path(folder).mkdir(exist_ok=True)
    path = f"{folder}/{time.strftime('%Y%m%d_%H%M%S')}_{name}.png"
    cv2.imwrite(path, img)
    return path


def log_run(result: str, lvl) -> None:
    new = not Path("runs.csv").exists()
    with open("runs.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "result", "taps", "draws", "failed_taps"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), result, lvl.taps, lvl.draws, lvl.failed_taps])


class Level:
    """What the bot remembers during one level."""

    def __init__(self, copies: int, pile_ref):
        self.mem = LevelMemory(copies)
        self.pile_ref = pile_ref      # look of the draw pile at level start (None = pile not seen)
        self.taps = self.draws = self.failed_taps = 0
        self.ended_by_me = False
        self.last_view = None
        self.last_chain_len = 0
        self.chain = []               # taps made last step, verified on the next screenshot
        self.drew = False
        self.prev_active = None
        self.draw_fails = 0           # draws in a row that didn't change the bottom card
        self.pile_unsure = 0
        self.suspects = []            # (x, y, rank) of cards whose tap did nothing
        self.suspect_n = None         # board size when the suspects were recorded
        self.seen_keys = set()        # unreadable cards already saved / asked about
        self.gambled = set()
        self.wild_drawn = False       # we just drew a WILD from the pile
        self.plus5_used = 0           # free +5 tiles used this level
        self.end_taps = 0             # END LEVEL taps that didn't end the level

    def is_suspect(self, c) -> bool:
        return any(abs(c.x - x) < 15 and abs(c.y - y) < 15 and c.rank == r for x, y, r in self.suspects)


def key(c):
    return (c.x // 20, c.y // 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--serial")
    ap.add_argument("--levels", type=int, default=0, help="stop after this many wins (0 = never)")
    ap.add_argument("--max-steps", type=int, default=100000)
    args = ap.parse_args()

    cfg = json.load(open("config.json"))
    phone = Phone(args.serial)
    book = GlyphBook(cfg["glyph_dir"], cfg["max_glyph_dist"], cfg["max_glyph_ratio"])
    if book.missing_ranks():
        print("No labelled samples yet for:", ", ".join(NAMES[r] for r in book.missing_ranks()))
        if not args.learn:
            print("Run 'python main.py --learn' until every rank is labelled, or those cards will be skipped.")
    if not args.dry_run:
        try:
            phone.keep_awake()
        except Exception:
            pass
    tap = (lambda x, y: None) if args.dry_run else phone.tap
    screens = ScreenBook(cfg.get("screen_dir", "screens"))

    lvl = None
    wins = losses_row = unknown = bad_active = no_active = 0

    for step in range(args.max_steps):
        img = phone.stable_grab()
        scene, btn = detect_scene(img)

        # ---------------- outside a level ----------------
        if scene != "GAME":
            if lvl is not None:
                v = lvl.last_view
                cleared = v is not None and v.blue_area < 1500 and len(v.board) <= lvl.last_chain_len
                result = "loss" if lvl.ended_by_me else ("win" if cleared else "unknown")
                log_run(result, lvl)
                print(f"== level over: {result} (taps={lvl.taps}, draws={lvl.draws}, failed taps={lvl.failed_taps})")
                if result == "win":
                    wins, losses_row = wins + 1, 0
                else:
                    losses_row += 1
                lvl = None
                if args.levels and wins >= args.levels:
                    print(f"Reached {wins} wins. Stopping.")
                    break
            if scene == "BUTTON":
                unknown = 0
                if losses_row >= cfg["max_losses_in_a_row"]:
                    print(f"{losses_row} levels in a row not won (each try costs coins). Stopping.")
                    break
                if not cfg["auto_start_levels"]:
                    print("At a level popup and auto_start_levels is off. Stopping.")
                    break
                print(f"tapping the {btn[2]} button")
                tap(btn[0], btn[1])
                time.sleep(2.5)
            elif (hit := screens.match(img)) is not None:
                meta, pos = hit
                unknown = 0
                if meta["action"] == "stop":
                    print("taught screen: stop here. Stopping.")
                    break
                if meta["action"] == "tap":
                    print(f"known screen, tapping at {pos}")
                    tap(*pos)
                else:
                    print("known screen, waiting for it to go away")
                time.sleep(2.0)
            else:
                unknown += 1
                print(f"unknown screen ({unknown}), not tapping anything")
                if args.learn and unknown >= 2:
                    meta = screens.teach(img)
                    if meta:
                        print(f"  learned this screen ({meta['action']})")
                        unknown = 0
                    continue
                if unknown == 3:
                    print(f"  saved {save('unknown', img, 'screen')}. Send it to me so the bot can learn this screen.")
                if unknown >= cfg["max_unknown_screens"]:
                    print("Stuck on an unknown screen. Stopping.")
                    break
                time.sleep(1.5)
            if args.dry_run:
                break
            continue

        # ---------------- in a level ----------------
        unknown = 0
        strict = args.learn and bool(book.missing_ranks())
        view = read_game(img, book, strict)
        a = view.active
        if a is None and view.active_gold and ((lvl and lvl.wild_drawn) or no_active >= 3):
            a = view.active = wild_card(img)          # a WILD is the bottom card: anything can be played
        if a is None:
            no_active += 1
            print("bottom card not found (animation?), waiting")
            time.sleep(0.5)
            continue
        no_active = 0
        if a.rank is None:
            if args.learn:
                r = ask_rank(a.crop, "Rank of the BOTTOM card?")
                if r:
                    book.add(r, a.crop)
                continue
            bad_active += 1
            if bad_active >= 5:
                save("pending", a.crop, f"active_{NAMES.get(a.read.guess, 'x')}")
                print("Can't read the bottom card. Run 'python main.py --learn' to label it. Stopping.")
                break
            time.sleep(0.5)
            continue
        bad_active = 0

        if lvl is None:
            lvl = Level(cfg["copies_per_rank"], None)
            if a.rank != WILD:
                lvl.mem.see_drawn(a.rank)
            print("== new level")
        else:
            # Check that the previous step's actions actually happened.
            if lvl.drew:
                if lvl.prev_active.shape != a.crop.shape or image_diff(lvl.prev_active, a.crop) > 6:
                    if a.rank != WILD:
                        lvl.mem.see_drawn(a.rank)
                    lvl.draw_fails = 0
                else:
                    lvl.draw_fails += 1
                    print("  the draw didn't change the bottom card")
            if lvl.chain:
                bad = first_failed_tap(lvl.chain, view.board)
                if bad:
                    lvl.failed_taps += 1
                    print(f"  tapping {NAMES[bad.rank]} at ({bad.x},{bad.y}) did nothing: misread or covered, skipping it")
                    lvl.suspects.append((bad.x, bad.y, bad.rank))
                    lvl.suspect_n = len(view.board)
                    card = next((c for c in view.board if abs(c.x - bad.x) < 12 and abs(c.y - bad.y) < 12), None)
                    if card is not None:
                        if args.learn:
                            r = ask_rank(card.crop, f"Tapping this card did nothing. Bot read it as {NAMES[bad.rank]}."
                                                    " Real rank?", allow_keep=True)
                            if r and r != "keep" and r != bad.rank:
                                book.add(r, card.crop)
                                print("  thanks, learned")
                                lvl.chain, lvl.drew = [], False
                                continue
                        else:
                            save("pending", card.crop, f"failed_{NAMES[bad.rank]}")
        lvl.chain, lvl.drew, lvl.prev_active = [], False, a.crop
        if a.rank != WILD:
            lvl.wild_drawn = False
        if lvl.pile_ref is None and not view.end_level and not view.pile_wild:
            lvl.pile_ref = view.pile      # remember the normal card back of the draw pile

        # Cards the bot isn't sure about: ask (learn mode) or keep for later labelling.
        fresh = [c for c in view.unreadable if key(c) not in lvl.seen_keys]
        for c in fresh:
            lvl.seen_keys.add(key(c))
            guess = NAMES.get(c.read.guess, "?")
            if args.learn:
                r = ask_rank(c.crop, f"Rank of this board card? (bot's guess: {guess})")
                if r:
                    book.add(r, c.crop)
            else:
                save("pending", c.crop, f"guess_{guess}")
        if fresh and args.learn:
            continue

        for c in view.board:
            lvl.mem.see_board(c.x, c.y, c.rank)
        if lvl.suspects and len(view.board) != lvl.suspect_n:
            lvl.suspects.clear()          # the board changed, a covered card may be free now

        playable = [c for c in view.board if not lvl.is_suspect(c)]
        free = [Free((c.x, c.y), c.rank, c.unlock, c.x, c.y) for c in playable]
        chain = plan(a.rank, free, lvl.mem.probs(), None, cfg["wrap_around"])
        lvl.last_view = view

        names = lambda cs: " ".join(NAMES[c.rank] for c in cs) or "-"
        note = (f"[{step}] bottom={NAMES[a.rank]} board=[{names(view.board)}] unsure={len(view.unreadable)} "
                f"skipped={len(view.board) - len(playable)} plan=[{names(chain)}]")
        print(note)
        if args.debug:
            save("debug", annotate(img, view, chain, lvl.is_suspect, note), f"step{step}")
            save("debug", img, f"step{step}_raw")

        if chain:
            for c in chain:
                tap(c.x, c.y)
                time.sleep(cfg["tap_delay"])
            lvl.chain, lvl.taps, lvl.last_chain_len, lvl.pile_unsure = chain, lvl.taps + len(chain), len(chain), 0
        elif not view.board and not view.unreadable and view.blue_area < 1500:
            print("  board cleared, waiting for the win screen")
            time.sleep(1.0)
        else:
            lvl.last_chain_len = 0
            pile_ok = (not view.end_level and lvl.draw_fails < 2
                       and lvl.pile_ref is not None and pile_matches(lvl.pile_ref, view.pile))
            if view.pile_wild and lvl.draw_fails < 2:
                print("  no moves -> taking the free WILD card on top of the draw pile")
                tap(*G.DRAW_TAP)
                lvl.drew, lvl.wild_drawn, lvl.draws, lvl.pile_unsure = True, True, lvl.draws + 1, 0
            elif pile_ok:
                print("  no moves -> drawing")
                tap(*G.DRAW_TAP)
                lvl.drew, lvl.draws, lvl.pile_unsure = True, lvl.draws + 1, 0
            elif view.end_level:
                # The pile is empty. Taps on cards are free, so try the cards it couldn't read first.
                gamble = [c for c in view.unreadable + [c for c in view.board if lvl.is_suspect(c)]
                          if key(c) not in lvl.gambled]
                if gamble:
                    lvl.gambled.add(key(gamble[0]))
                    print("  pile empty, trying an uncertain card (free)")
                    tap(gamble[0].x, gamble[0].y)
                elif view.free_plus5 and lvl.plus5_used < cfg["max_free_plus5_per_level"]:
                    print("  pile empty, you own a free +5 (red badge) -> using it")
                    tap(*G.PLUS5_TAP)
                    lvl.plus5_used += 1
                    lvl.draw_fails, lvl.pile_ref, lvl.gambled = 0, None, set()  # new cards: re-learn the pile look
                    time.sleep(1.5)
                else:
                    if lvl.end_taps >= 2:
                        print(f"  END LEVEL didn't end the level. Saved {save('unknown', img, 'end_level')}."
                              " Send it to me. Stopping.")
                        break
                    print("  no moves and the draw pile is empty -> END LEVEL (never buying +5 cards)")
                    lvl.ended_by_me = True
                    lvl.end_taps += 1
                    tap(*G.END_LEVEL_TAP)
                    time.sleep(2.0)
            else:
                lvl.pile_unsure += 1
                print("  the draw pile doesn't look like at level start, not tapping it (could be the +5 shop tile)")
                if lvl.pile_unsure >= 3:
                    print(f"  saved {save('unknown', img, 'pile')}. Send it to me. Stopping to be safe.")
                    break
                time.sleep(1.0)
        if args.dry_run:
            break


if __name__ == "__main__":
    main()
