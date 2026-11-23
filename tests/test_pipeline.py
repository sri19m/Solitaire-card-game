"""Synthetic-frame tests (no phone needed). Real-screen accuracy still has to be checked on the phone."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot import geometry as G
from bot.brain import Free, LevelMemory, adjacent, first_failed_tap, plan, rank_probs
from bot.vision import NAMES, GlyphBook, detect_scene, find_cards, pile_matches, read_game

WOOD = (60, 165, 235)      # BGR orange table
BACK = (230, 120, 40)      # BGR blue card back
CREAM = (238, 244, 247)


def card_img(rank, color=(70, 40, 20)):
    c = np.full((138, 89, 3), 255, np.uint8)
    cv2.rectangle(c, (3, 3), (85, 134), CREAM, -1)
    cv2.putText(c, NAMES[rank], (7, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    big = 1.6 if rank == 10 else 2.6
    cv2.putText(c, NAMES[rank], (8 if rank == 10 else 14, 112), cv2.FONT_HERSHEY_DUPLEX, big, color, 4)
    return c


def paste(img, card, x, y, outline=True, angle=0.0):
    h, w = card.shape[:2]
    if outline:
        card = card.copy()
        cv2.rectangle(card, (0, 0), (w - 1, h - 1), (170, 170, 170), 1)
    if angle:
        pad = 30
        big = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.uint8)
        m = np.zeros(big.shape[:2], np.uint8)
        big[pad:pad + h, pad:pad + w] = card
        m[pad:pad + h, pad:pad + w] = 255
        R = cv2.getRotationMatrix2D((w / 2 + pad, h / 2 + pad), angle, 1.0)
        big = cv2.warpAffine(big, R, big.shape[1::-1])
        m = cv2.warpAffine(m, R, m.shape[1::-1]) > 200
        region = img[y - pad:y - pad + big.shape[0], x - pad:x - pad + big.shape[1]]
        region[m] = big[m]
    else:
        img[y:y + h, x:x + w] = card


def frame(board=(), active=None, backs=()):
    img = np.zeros((716, 1536, 3), np.uint8)
    img[:] = WOOD
    for bx, by in backs:
        img[by:by + 138, bx:bx + 89] = 255
        img[by + 3:by + 135, bx + 3:bx + 86] = BACK
    for x, y, r in board:
        paste(img, card_img(r, (40, 30, 200) if r % 2 else (70, 40, 20)), x, y)
    if active:
        paste(img, card_img(active), 814, 553)
    img[G.PILE_TOP_ROI[1]:G.PILE_TOP_ROI[1] + 120, G.PILE_TOP_ROI[0]:G.PILE_TOP_ROI[0] + 70] = BACK
    return img


def full_book(folder, ranks=range(1, 14)):
    book = GlyphBook(folder)
    for r in ranks:
        for col in ((70, 40, 20), (40, 30, 200)):
            img = frame()
            paste(img, card_img(r, col), 814, 553)
            book.add(r, find_cards(img, G.ACTIVE_ROI)[0][3])
    return book


def test_read_cards_and_counts():
    with tempfile.TemporaryDirectory() as d:
        book = full_book(d)
        spots = [(369, 245), (537, 305), (723, 370), (910, 305), (1078, 245)]
        ranks = [11, 7, 9, 5, 8]
        v = read_game(frame([(x, y, r) for (x, y), r in zip(spots, ranks)], 1, [(369, 125), (489, 125)]), book)
        assert v.active.rank == 1
        assert sorted(c.rank for c in v.board) == sorted(ranks)
        assert not v.unreadable and v.blue_area > 5000 and not v.end_level


def test_tilted_card():
    with tempfile.TemporaryDirectory() as d:
        book = full_book(d)
        img = frame(active=3)
        paste(img, card_img(9, (40, 30, 200)), 600, 250, angle=9)
        v = read_game(img, book)
        assert [c.rank for c in v.board] == [9]


def test_overlapping_face_up_cards_only_top_is_free():
    with tempfile.TemporaryDirectory() as d:
        book = full_book(d)
        img = frame(active=3)
        paste(img, card_img(7), 500, 250, outline=False)
        paste(img, card_img(8), 545, 262, outline=False)   # 8 lies on top of the 7
        v = read_game(img, book)
        assert [c.rank for c in v.board] == [8]


def test_missing_rank_is_not_guessed_in_strict_mode():
    # The bug from the first real run: no 5 labelled, so a 5 was read as a 4.
    with tempfile.TemporaryDirectory() as d:
        book = full_book(d, ranks=[r for r in range(1, 14) if r != 5])
        img = frame([(500, 250, 5)], 3)
        v = read_game(img, book, strict=True)
        assert not v.board and len(v.unreadable) == 1


def test_scene_whitelist():
    img = frame(active=5)
    img[515:615, 1415:1495] = (30, 190, 255)                          # gold wild card
    assert detect_scene(img)[0] == "GAME"
    popup = np.full((716, 1536, 3), (60, 20, 30), np.uint8)
    cv2.rectangle(popup, (655, 450), (880, 550), (40, 200, 60), -1)  # green PLAY
    assert detect_scene(popup)[0] == "UNKNOWN"                        # no red X: not the level popup
    cv2.circle(popup, (1083, 125), 25, (40, 40, 230), -1)            # red X
    scene, btn = detect_scene(popup)
    assert scene == "BUTTON" and abs(btn[0] - 767) < 10 and abs(btn[1] - 500) < 10
    shop = np.full((716, 1536, 3), (60, 20, 30), np.uint8)
    cv2.rectangle(shop, (950, 560), (1250, 660), (40, 200, 60), -1)  # some other green button
    assert detect_scene(shop)[0] == "UNKNOWN"


def test_end_level_and_pile_checks():
    img = frame(active=5)
    with tempfile.TemporaryDirectory() as d:
        v = read_game(img, GlyphBook(d))
        assert not v.end_level and pile_matches(v.pile, v.pile)
        shop = img.copy()
        x, y, w, h = G.PILE_TOP_ROI
        cv2.putText(shop, "+5", (x + 5, y + 75), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 6)
        assert not pile_matches(v.pile, read_game(shop, GlyphBook(d)).pile)
        cv2.rectangle(img, (280, 615), (480, 680), (40, 40, 230), -1)  # red END LEVEL
        assert read_game(img, GlyphBook(d)).end_level


def test_failed_tap_detection():
    chain = [Free(0, 4, 0, 500, 300), Free(1, 5, 0, 700, 300)]
    assert first_failed_tap(chain, [Free(9, 4, 0, 503, 298)]).rank == 4
    assert first_failed_tap(chain, [Free(9, 9, 0, 503, 298)]) is None


def test_planning_and_wrap():
    assert adjacent(13, 1) and adjacent(1, 13) and not adjacent(5, 7)
    free = [Free(i, r) for i, r in enumerate([6, 7, 8, 2])]
    assert [f.rank for f in plan(5, free, {})] == [6, 7, 8]
    assert plan(5, [Free(0, 9)], {}) == []
    assert [f.rank for f in plan(13, [Free(0, 1), Free(1, 2)], {})] == [1, 2]


def test_card_counting():
    m = LevelMemory()
    for r in (7, 7, 7, 7):
        m.see_drawn(r)
    m.see_board(100, 100, 3)
    m.see_board(105, 98, 3)
    p = m.probs()
    assert p[7] == 0 and p[3] > 0 and abs(sum(p.values()) - 1) < 1e-9
    assert sum(m.seen().values()) == 5
    assert rank_probs({}, 4)[1] == 1 / 13


def test_glyph_self_check():
    with tempfile.TemporaryDirectory() as d:
        res = full_book(d).leave_one_out()
        assert all(rd.guess == r for _, r, rd in res)


GOLD = (40, 200, 250)  # BGR of the golden WILD card


def test_free_wild_on_draw_pile_is_recognised():
    img = frame(active=7)
    x, y, w, h = G.PILE_TOP_ROI
    img[y - 8:y + h + 8, x - 6:x + w + 6] = GOLD
    with tempfile.TemporaryDirectory() as d:
        v = read_game(img, GlyphBook(d))
        assert v.pile_wild and not v.end_level
        cv2.rectangle(img, (280, 615), (480, 680), (40, 40, 230), -1)   # END LEVEL shown -> never draw
        assert not read_game(img, GlyphBook(d)).pile_wild
        assert not read_game(frame(active=7), GlyphBook(d)).pile_wild    # normal card back


def test_wild_as_bottom_card():
    from bot.brain import WILD
    img = frame([(500, 250, 9), (700, 250, 10)])
    img[553:691, 814:903] = GOLD
    with tempfile.TemporaryDirectory() as d:
        v = read_game(img, full_book(d))
        assert v.active is None and v.active_gold
    chain = plan(WILD, [Free(0, 9), Free(1, 10), Free(2, 4)], {})
    assert sorted(c.rank for c in chain) == [9, 10]   # starts anywhere, then follows the rules


def test_taught_screens():
    from bot.screens import ScreenBook
    congrats = np.full((716, 1536, 3), (90, 30, 50), np.uint8)
    cv2.putText(congrats, "CONGRATULATIONS!", (390, 180), cv2.FONT_HERSHEY_DUPLEX, 2.5, (40, 200, 250), 8)
    cv2.putText(congrats, "TAP TO COLLECT", (670, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    with tempfile.TemporaryDirectory() as d:
        book = ScreenBook(d)
        assert book.match(congrats) is None
        book.remember(congrats, "tap", (767, 670))
        noisy = cv2.add(congrats, np.random.default_rng(0).integers(0, 12, congrats.shape, dtype=np.uint8))
        meta, pos = ScreenBook(d).match(noisy)                 # reloaded from disk
        assert meta["action"] == "tap" and pos == (767, 670)
        other = congrats.copy()
        cv2.rectangle(other, (600, 640), (940, 700), (40, 200, 60), -1)  # e.g. a green buy button there
        assert book.match(other) is None
        assert book.match(frame(active=5)) is None
