"""Play loop.

    python main.py --learn        # first run: asks you to label cards it can't read yet
    python main.py                # autonomous play
    python main.py --dry-run      # read the screen and print decisions, never tap
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
from bot.brain import Free, LevelMemory, plan
from bot.learn import ask_rank
from bot.vision import GlyphBook, detect_scene, read_game


def log_run(result: str, moves: int, draws: int):
    new = not Path("runs.csv").exists()
    with open("runs.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "result", "taps", "draws"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), result, moves, draws])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--serial")
    ap.add_argument("--levels", type=int, default=0, help="stop after this many wins (0 = never)")
    ap.add_argument("--max-steps", type=int, default=100000)
    args = ap.parse_args()

    cfg = json.load(open("config.json"))
    phone = Phone(args.serial)
    if not args.dry_run:
        phone.keep_awake()
    book = GlyphBook(cfg["glyph_dir"])
    tap = (lambda x, y: None) if args.dry_run else phone.tap

    mem = None                 # LevelMemory while a level is in progress
    last = None                # last GameView of the level
    last_plan_len = 0
    taps = draws = 0
    just_drew = False
    ended_by_me = False
    wins = losses_row = unknown = 0
    still_draws = 0            # draws that didn't change the active card
    prev_active = None

    for step in range(args.max_steps):
        img = phone.stable_grab()
        scene, btn = detect_scene(img)

        # ---------------- not in a level ----------------
        if scene != "GAME":
            if mem is not None:
                cleared = last is not None and last.blue_area < 1500 and len(last.board) <= last_plan_len
                result = "win" if cleared and not ended_by_me else "loss"
                log_run(result, taps, draws)
                print(f"== level over: {result} (taps={taps}, draws={draws})")
                if result == "win":
                    wins, losses_row = wins + 1, 0
                else:
                    losses_row += 1
                mem, last, taps, draws, ended_by_me, still_draws = None, None, 0, 0, False, 0
                if args.levels and wins >= args.levels:
                    print(f"Reached {wins} wins. Stopping.")
                    break
            if scene == "BUTTON":
                unknown = 0
                if losses_row >= cfg["max_losses_in_a_row"] or (losses_row and not cfg["retry_on_loss"]):
                    print(f"Lost {losses_row} in a row (each retry costs coins). Stopping.")
                    break
                print(f"green button at {btn}, tapping")
                tap(*btn)
                time.sleep(2.0)
            else:
                unknown += 1
                print(f"unknown screen ({unknown})")
                if unknown == 3:
                    Path("unknown").mkdir(exist_ok=True)
                    p = f"unknown/{time.strftime('%H%M%S')}.png"
                    cv2.imwrite(p, img)
                    print(f"  saved {p}. Send it to me and I'll teach the bot this screen.")
                if unknown >= cfg["max_unknown_screens"]:
                    print("Stuck on an unknown screen. Stopping.")
                    break
                time.sleep(1.5)
            if args.dry_run:
                break
            continue

        # ---------------- in a level ----------------
        unknown = 0
        view = read_game(img, book)

        if view.unreadable and args.learn:
            for c in view.unreadable + ([view.active] if view.active and view.active.rank is None else []):
                r = ask_rank(c.crop)
                if r:
                    book.add(r, c.crop)
            continue
        if view.active is not None and view.active.rank is None and args.learn:
            r = ask_rank(view.active.crop)
            if r:
                book.add(r, view.active.crop)
            continue
        if view.active is None or view.active.rank is None:
            print("can't read the active card, waiting")
            time.sleep(1)
            continue

        if mem is None:
            mem = LevelMemory(cfg["copies_per_rank"])
            mem.see_drawn(view.active.rank)
            print("== new level")
        elif just_drew:
            same = prev_active is not None and prev_active.shape == view.active.crop.shape and \
                float(cv2.absdiff(prev_active, view.active.crop).mean()) < 3.0
            still_draws = still_draws + 1 if same else 0
            if not same:
                mem.see_drawn(view.active.rank)
        just_drew = False
        prev_active = view.active.crop
        if still_draws >= 2:      # drawing no longer changes the card: the pile is empty
            view.stock_empty = True
        for c in view.board:
            mem.see_board(c.x, c.y, c.rank)
        last = view

        free = [Free((c.x, c.y), c.rank, c.unlock, c.x, c.y) for c in view.board]
        chain = plan(view.active.rank, free, mem.probs(), None, cfg["wrap_around"])
        names = lambda cs: [c.rank for c in cs]
        print(f"[{step}] active={view.active.rank} free={names(view.board)} "
              f"unreadable={len(view.unreadable)} plan={names(chain)}")

        if chain:
            still_draws = 0
            last_plan_len = len(chain)
            for c in chain:
                tap(c.x, c.y)
                taps += 1
                time.sleep(cfg["tap_delay"])
        elif view.stock_empty and view.unreadable:
            print("  pile empty but some cards are unreadable; not ending the level")
            time.sleep(1)
        elif view.stock_empty:
            print("  no moves and the draw pile is empty -> END LEVEL")
            ended_by_me = True
            tap(*G.END_LEVEL_TAP)
            time.sleep(2.0)
        else:
            last_plan_len = 0
            print("  no moves -> drawing")
            tap(*G.DRAW_TAP)
            draws += 1
            just_drew = True
        if args.dry_run:
            break


if __name__ == "__main__":
    main()
