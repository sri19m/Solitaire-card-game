"""Synthetic-frame tests (no phone needed). Real-screen accuracy still has to be checked with real screenshots."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.brain import Free, LevelMemory, adjacent, plan, rank_probs
from bot.vision import GlyphBook, NAMES, detect_scene, read_game

WOOD = (60, 165, 235)      # BGR, orange table
BACK = (230, 120, 40)      # BGR, blue card back
CREAM = (238, 244, 247)


def draw_face(img, x, y, rank, color=(70, 40, 20)):
    cv2.rectangle(img, (x, y), (x + 89, y + 138), (255, 255, 255), -1)
    cv2.rectangle(img, (x + 3, y + 3), (x + 86, y + 135), CREAM, -1)
    cv2.putText(img, NAMES[rank], (x + 6, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # corner index
    cv2.putText(img, NAMES[rank], (x + 12 if rank != 10 else x + 10, y + 105), cv2.FONT_HERSHEY_DUPLEX,
                2.6 if rank != 10 else 1.6, color, 4)


def draw_back(img, x, y):
    cv2.rectangle(img, (x, y), (x + 89, y + 138), (255, 255, 255), -1)
    cv2.rectangle(img, (x + 3, y + 3), (x + 86, y + 135), BACK, -1)


def frame(board, active, backs=()):
    img = np.zeros((716, 1536, 3), np.uint8)
    img[:] = WOOD
    for bx, by in backs:
        draw_back(img, bx, by)
    for (x, y, r) in board:
        draw_face(img, x, y, r, (40, 30, 200) if r % 2 else (70, 40, 20))
    draw_face(img, 814, 553, active)
    return img


def book_with_all_ranks(folder):
    book = GlyphBook(folder)
    for r in range(1, 14):
        img = frame([], r)
        c = read_game(img, book).active  # unreadable at first; crop still returned
        book.add(r, c.crop)
    return book


def test_read_cards():
    with tempfile.TemporaryDirectory() as d:
        book = book_with_all_ranks(d)
        spots = [(369, 245), (537, 305), (723, 370), (910, 305), (1078, 245)]
        ranks = [11, 7, 9, 9, 10]
        img = frame([(x, y, r) for (x, y), r in zip(spots, ranks)], 1, backs=[(369, 125), (489, 125)])
        v = read_game(img, book)
        assert v.active.rank == 1
        assert sorted(c.rank for c in v.board) == sorted(ranks)
        assert not v.unreadable
        assert v.blue_area > 5000


def test_unknown_card_is_flagged_not_guessed():
    with tempfile.TemporaryDirectory() as d:
        book = GlyphBook(d)
        book.add(7, read_game(frame([], 7), book).active.crop)
        v = read_game(frame([(400, 250, 12)], 7), book)
        assert v.active.rank == 7 and len(v.unreadable) == 1


def test_scene_game_and_button():
    img = frame([], 5)
    cv2.rectangle(img, (1415, 515), (1495, 615), (30, 190, 255), -1)  # gold wild card
    assert detect_scene(img)[0] == "GAME"
    popup = np.zeros((716, 1536, 3), np.uint8)
    popup[:] = (60, 20, 30)
    cv2.rectangle(popup, (655, 450), (880, 550), (40, 200, 60), -1)   # green PLAY
    scene, pos = detect_scene(popup)
    assert scene == "BUTTON" and abs(pos[0] - 767) < 10 and abs(pos[1] - 500) < 10


def test_undo_button_is_never_tapped():
    img = np.zeros((716, 1536, 3), np.uint8)
    cv2.rectangle(img, (950, 560), (1050, 650), (40, 200, 60), -1)    # green undo
    assert detect_scene(img)[0] == "UNKNOWN"


def test_planning_and_wrap():
    assert adjacent(13, 1) and adjacent(1, 13) and not adjacent(5, 7)
    free = [Free(i, r) for i, r in enumerate([6, 7, 8, 2])]
    assert [f.rank for f in plan(5, free, {})] == [6, 7, 8]
    assert plan(5, [Free(0, 9)], {}) == []


def test_card_counting():
    m = LevelMemory()
    for r in (7, 7, 7, 7):
        m.see_drawn(r)
    m.see_board(100, 100, 7 if False else 3)
    m.see_board(105, 98, 3)            # same card seen again: not double counted
    p = m.probs()
    assert p[7] == 0 and p[3] > 0
    assert abs(sum(p.values()) - 1) < 1e-9
    assert sum(m.seen().values()) == 5
    assert rank_probs({}, 4)[1] == 1 / 13
