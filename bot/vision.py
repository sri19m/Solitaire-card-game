"""Screen understanding: scene detection, card finding, rank recognition.

Rank recognition needs no pre-made templates. The bot cuts out each card, and a
nearest-neighbour matcher compares the big centre glyph with cards you labelled once
(glyphs/ folder). Suits are ignored.
"""
import glob
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as G

RANKS = {"A": 1, "J": 11, "Q": 12, "K": 13, **{str(n): n for n in range(2, 11)}}
NAMES = {v: k for k, v in RANKS.items()}


# ---------- colour masks ----------
def cream_mask(hsv):
    return ((hsv[..., 1] < 50) & (hsv[..., 2] > 200)).astype(np.uint8) * 255


def blue_mask(hsv):
    return ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 125) & (hsv[..., 1] > 110) & (hsv[..., 2] > 140)).astype(np.uint8) * 255


def green_mask(hsv):
    return ((hsv[..., 0] >= 45) & (hsv[..., 0] <= 80) & (hsv[..., 1] > 150) & (hsv[..., 2] > 120)).astype(np.uint8) * 255


def gold_fraction(img, roi, hmin=15, hmax=35):
    x, y, w, h = roi
    hsv = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= hmin) & (hsv[..., 0] <= hmax) & (hsv[..., 1] > 150) & (hsv[..., 2] > 200)
    return float(m.mean())


def blue_fraction(img, roi):
    x, y, w, h = roi
    return float((blue_mask(cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)) > 0).mean())


# ---------- scene ----------
def _inside(pt, zone):
    x, y, w, h = zone
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


def detect_scene(img):
    """Return ("GAME", None) | ("BUTTON", (x, y)) | ("UNKNOWN", None).

    BUTTON means a big green button (Play / Try again / Level / Continue) is on a non-game screen.
    """
    in_game = gold_fraction(img, G.WILD_ROI) > 0.25 or blue_fraction(img, G.GEAR_ROI) > 0.25
    if in_game:
        return "GAME", None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(green_mask(hsv), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        c = (int(cent[i][0]), int(cent[i][1]))
        if area >= 6000 and not _inside(c, G.UNDO_ZONE) and (best is None or area > best[0]):
            best = (area, c)
    return ("BUTTON", best[1]) if best else ("UNKNOWN", None)


def stock_empty(img) -> bool:
    """When the draw pile runs out, a '+5 cards' shop tile with a coin appears in its place."""
    return gold_fraction(img, G.STOCK_COIN_ROI, 20, 32) > 0.12


# ---------- cards ----------
@dataclass
class Card:
    rank: int | None
    x: int
    y: int
    box: tuple
    crop: np.ndarray
    unlock: float = 0.0


def find_card_boxes(img, roi):
    """Boxes (x, y, w, h) of fully visible face-up cards inside roi (absolute coordinates)."""
    rx, ry, rw, rh = roi
    hsv = cv2.cvtColor(img[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cream_mask(hsv), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # outer shape, glyph holes filled
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 70 <= w <= 105 and 110 <= h <= 150 and cv2.contourArea(c) > 0.85 * w * h:
            boxes.append((rx + x, ry + y, w, h))
    return sorted(boxes, key=lambda b: (b[1] // 40, b[0]))


def glyph_feature(card: np.ndarray):
    """Binary image of the big centre glyph, normalised. Returns (vector, aspect) or None."""
    h, w = card.shape[:2]
    reg = card[int(.40 * h):int(.90 * h), int(.10 * w):int(.90 * w)]
    bg = np.median(reg.reshape(-1, 3), axis=0)
    ink = (np.linalg.norm(reg.astype(np.float32) - bg, axis=2) > 80).astype(np.uint8)
    ys, xs = np.nonzero(ink)
    if len(ys) < 30:
        return None
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    vec = cv2.resize(ink.astype(np.float32), (24, 32), interpolation=cv2.INTER_AREA).ravel()
    return vec, ink.shape[1] / ink.shape[0]


class GlyphBook:
    """Labelled card crops on disk: glyphs/<RANK>_<n>.png"""

    def __init__(self, folder: str, max_dist: float = 0.22):
        self.folder, self.max_dist = folder, max_dist
        self.samples: list[tuple[int, np.ndarray, float]] = []
        os.makedirs(folder, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
            name = re.split(r"[_.]", os.path.basename(path))[0].upper()
            img = cv2.imread(path)
            f = glyph_feature(img) if img is not None and name in RANKS else None
            if f:
                self.samples.append((RANKS[name], f[0], f[1]))

    def known_ranks(self) -> set[int]:
        return {r for r, _, _ in self.samples}

    def classify(self, crop: np.ndarray):
        f = glyph_feature(crop)
        if not f or not self.samples:
            return None, 1.0
        best_r, best_d = None, 9.0
        for r, vec, asp in self.samples:
            d = float(np.abs(vec - f[0]).mean()) + 0.3 * abs(asp - f[1])
            if d < best_d:
                best_r, best_d = r, d
        return (best_r, best_d) if best_d <= self.max_dist else (None, best_d)

    def add(self, rank: int, crop: np.ndarray) -> None:
        n = 0
        while os.path.exists(os.path.join(self.folder, f"{NAMES[rank]}_{n}.png")):
            n += 1
        cv2.imwrite(os.path.join(self.folder, f"{NAMES[rank]}_{n}.png"), crop)
        f = glyph_feature(crop)
        if f:
            self.samples.append((rank, f[0], f[1]))


@dataclass
class GameView:
    active: Card | None
    board: list[Card]
    unreadable: list[Card]   # face-up cards whose rank wasn't recognised
    blue_area: int           # face-down pixels left on the board
    stock_empty: bool


def read_game(img, book: GlyphBook) -> GameView:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = blue_mask(hsv)
    pad = 22
    cards, bad = [], []
    for (x, y, w, h) in find_card_boxes(img, G.BOARD_ROI):
        crop = img[y:y + h, x:x + w]
        rank, _ = book.classify(crop)
        ring = blue[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        outside = (ring > 0).sum() - (blue[y:y + h, x:x + w] > 0).sum()
        unlock = min(outside / (0.6 * w * h), 2.0)
        c = Card(rank, x + w // 2, y + h // 2, (x, y, w, h), crop, float(unlock))
        (cards if rank else bad).append(c)

    active = None
    boxes = find_card_boxes(img, G.ACTIVE_ROI)
    if boxes:
        x, y, w, h = boxes[0]
        crop = img[y:y + h, x:x + w]
        rank, _ = book.classify(crop)
        active = Card(rank, x + w // 2, y + h // 2, (x, y, w, h), crop)

    bx, by, bw, bh = G.BOARD_ROI
    return GameView(active, cards, bad, int((blue[by:by + bh, bx:bx + bw] > 0).sum()), stock_empty(img))
