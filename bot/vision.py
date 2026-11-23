"""Screen understanding: which screen is shown, where the cards are, and each card's rank.

Rank reading needs no built-in images. You label cards once (python main.py --learn) and
GlyphBook compares new cards with your labelled samples. Suits are ignored.
"""
import glob
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as G

RANKS = {"A": 1, "J": 11, "Q": 12, "K": 13, **{str(n): n for n in range(2, 11)}}
NAMES = {v: k for k, v in RANKS.items()}
NAMES[0] = "WILD"
FACE_W, FACE_H = 90, 138          # every card crop is warped to this upright size


# ---------- colour masks (HSV, OpenCV hue is 0-179) ----------
def cream_mask(hsv):
    return ((hsv[..., 1] < 50) & (hsv[..., 2] > 200)).astype(np.uint8) * 255


def blue_mask(hsv):
    return ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 125) & (hsv[..., 1] > 110) & (hsv[..., 2] > 140)).astype(np.uint8) * 255


def green_mask(hsv):
    return ((hsv[..., 0] >= 40) & (hsv[..., 0] <= 80) & (hsv[..., 1] > 150) & (hsv[..., 2] > 120)).astype(np.uint8) * 255


def red_mask(hsv):
    h = hsv[..., 0]
    return (((h <= 8) | (h >= 170)) & (hsv[..., 1] > 140) & (hsv[..., 2] > 140)).astype(np.uint8) * 255


def gold_mask(hsv):
    return ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 35) & (hsv[..., 1] > 150) & (hsv[..., 2] > 200)).astype(np.uint8) * 255


def coin_mask(hsv):
    """Bright yellow of a coin icon (brighter and more yellow than the wooden table)."""
    return ((hsv[..., 0] >= 17) & (hsv[..., 0] <= 35) & (hsv[..., 1] > 120) & (hsv[..., 2] > 238)).astype(np.uint8) * 255


def fraction(mask_fn, img, roi) -> float:
    x, y, w, h = roi
    return float((mask_fn(cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)) > 0).mean())


# ---------- which screen ----------
def detect_scene(img):
    """Return (scene, button): scene is "GAME", "BUTTON" or "UNKNOWN".

    For "BUTTON", button is (x, y, name) of a whitelisted green button (Play / Try again /
    lobby Level). Any other green button (shops, offers, "continue for coins") is never returned.
    """
    if fraction(gold_mask, img, G.WILD_ROI) > 0.25 or fraction(blue_mask, img, G.GEAR_ROI) > 0.25:
        return "GAME", None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(green_mask(hsv), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 6000:
            continue
        cx, cy = cent[i]
        for name, (bx, by, r, needs_close) in G.SAFE_BUTTONS.items():
            if abs(cx - bx) <= r and abs(cy - by) <= r:
                if needs_close and fraction(red_mask, img, G.POPUP_CLOSE_ROI) < 0.3:
                    continue
                return "BUTTON", (int(cx), int(cy), name)
    return "UNKNOWN", None


# ---------- finding cards ----------
_DST = np.float32([[0, 0], [FACE_W - 1, 0], [FACE_W - 1, FACE_H - 1], [0, FACE_H - 1]])


def _corners(rect, ox, oy):
    """Corners (tl, tr, br, bl) of a card with its long side as the vertical axis, at any tilt.

    The card's 'up' direction is taken along the long side, pointing to the top of the screen.
    A card lying more than 90 degrees turned ends up upside down here; _upright() fixes that.
    """
    pts = cv2.boxPoints(rect)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    long_v, short_v = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    up = long_v / np.linalg.norm(long_v)
    if up[1] > 0:
        up = -up
    right = np.float32([-up[1], up[0]])
    c = np.float32(rect[0]) + np.float32([ox, oy])
    hl, hs = np.linalg.norm(long_v) / 2, np.linalg.norm(short_v) / 2
    return np.float32([c - right * hs + up * hl, c + right * hs + up * hl,
                       c + right * hs - up * hl, c - right * hs - up * hl])


def _upright(card):
    """Turn a straightened card 180 degrees if it is upside down.

    The big rank glyph always sits in the lower half of an upright card (the small corner index
    is at the top), so an upside-down 6 is never read as a 9.
    """
    h, w = card.shape[:2]
    my, mx = int(.04 * h), int(.06 * w)
    ink = _ink(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY))[my:h - my, mx:w - mx]
    n, _, st, cent = cv2.connectedComponentsWithStats(ink)
    big = [i for i in range(1, n) if st[i, 3] >= .18 * h]
    if not big:
        return card
    area = np.float32([st[i, 4] for i in big])
    cy = float((np.float32([cent[i][1] for i in big]) * area).sum() / area.sum())
    return card if cy >= 0.5 * ink.shape[0] else cv2.rotate(card, cv2.ROTATE_180)


def _as_card(img, contour, ox, oy, min_fill=0.75):
    """Upright crop of a card-shaped contour, or None. Handles slightly tilted cards and cards with a
    corner hidden by a board item. If such a card turns out not tappable, tap verification skips it."""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), _ = rect
    short, long_ = sorted((w, h))
    if not (G.CARD_SHORT[0] <= short <= G.CARD_SHORT[1]) or not (1.35 <= long_ / max(short, 1) <= 1.8):
        return None
    if cv2.contourArea(contour) < min_fill * w * h:
        return None
    quad = _corners(rect, ox, oy)
    crop = _upright(cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, _DST), (FACE_W, FACE_H)))
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    return int(cx + ox), int(cy + oy), (x, y, bw, bh), crop


def _split_merged(img, mask, box, rx, ry):
    """Overlapping face-up cards form one cream blob. Cut it along card outlines and keep the
    pieces that are still whole cards: those are the cards on top, the only tappable ones."""
    x, y, w, h = box
    gray = cv2.cvtColor(img[ry + y:ry + y + h, rx + x:rx + x + w], cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 15, 40), np.ones((3, 3), np.uint8))
    piece = cv2.bitwise_and(mask[y:y + h, x:x + w], cv2.bitwise_not(edges))
    piece = cv2.morphologyEx(piece, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = []
    for c in cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        card = _as_card(img, c, rx + x, ry + y, min_fill=0.75)
        if card:
            out.append(card)
    return out


def find_cards(img, roi):
    """Every fully visible face-up card in roi: [(cx, cy, box, upright_crop)] in screen coordinates."""
    rx, ry, rw, rh = roi
    hsv = cv2.cvtColor(img[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(cream_mask(hsv), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    card_area = G.CARD_W * G.CARD_H
    out = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.45 * card_area:
            continue
        card = _as_card(img, c, rx, ry)
        if card:
            out.append(card)
        elif w * h > 0.9 * card_area:
            # Not card-shaped: overlapping cards, or a white board item touching the card.
            # Cut along outlines first, then try stripping thin attachments.
            pieces = _split_merged(img, mask, (x, y, w, h), rx, ry)
            if not pieces:
                blob = np.zeros_like(mask)
                cv2.drawContours(blob, [c], -1, 255, -1)
                blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
                for c2 in cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                    piece = _as_card(img, c2, rx, ry)
                    if piece:
                        pieces.append(piece)
            out += pieces
    return sorted(out, key=lambda k: (k[1] // 40, k[0]))


# ---------- reading ranks ----------
def _ink(gray):
    return (gray < 175).astype(np.uint8)   # red and dark ink are both much darker than the cream face


def _glyph_box(ink, min_h):
    """Bounding box of the ink components at least min_h tall, ignoring specks touching the top/left edge."""
    n, _, st, _ = cv2.connectedComponentsWithStats(ink)
    boxes = [st[i, :4] for i in range(1, n)
             if st[i, 3] >= min_h and st[i, 4] >= 8 and st[i, 0] > 0 and st[i, 1] > 0]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes))


def _norm(g, size):
    return cv2.GaussianBlur(cv2.resize(g.astype(np.float32), size, interpolation=cv2.INTER_AREA), (3, 3), 0)


def glyph_feature(card):
    """Features of the big centre rank glyph and the small corner index. None if no glyph found."""
    card = _upright(cv2.resize(card, (FACE_W, FACE_H), interpolation=cv2.INTER_AREA))
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    ink = _ink(gray[int(.25 * FACE_H):int(.96 * FACE_H), int(.04 * FACE_W):int(.96 * FACE_W)])
    b = _glyph_box(ink, int(.18 * FACE_H))
    if b is None:
        return None
    g = ink[b[1]:b[3], b[0]:b[2]]
    corner = _ink(gray[int(.03 * FACE_H):int(.27 * FACE_H), int(.03 * FACE_W):int(.55 * FACE_W)])
    b2 = _glyph_box(corner, int(.07 * FACE_H))
    return {
        "c": _norm(g, (28, 36)),
        "a": g.shape[1] / g.shape[0],
        "k": _norm(corner[b2[1]:b2[3], b2[0]:b2[2]], (14, 18)) if b2 else None,
    }


def glyph_distance(f, s) -> float:
    d = float(np.abs(f["c"] - s["c"]).mean()) + 0.15 * abs(f["a"] - s["a"])
    if f["k"] is not None and s["k"] is not None:
        d = 0.65 * d + 0.35 * float(np.abs(f["k"] - s["k"]).mean())
    return d


@dataclass
class Read:
    rank: int | None      # accepted rank, None if not confident
    guess: int | None     # nearest rank even when not confident
    dist: float
    ratio: float          # best distance / best distance of any other rank (lower = more certain)


class GlyphBook:
    """Labelled card crops on disk: glyphs/<RANK>_<n>.png. Delete a file to remove a bad label."""

    def __init__(self, folder: str, max_dist: float = 0.16, max_ratio: float = 0.8):
        self.folder, self.max_dist, self.max_ratio = folder, max_dist, max_ratio
        self.samples: list[tuple[int, dict, str]] = []
        os.makedirs(folder, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
            name = re.split(r"[_.]", os.path.basename(path))[0].upper()
            img = cv2.imread(path) if name in RANKS else None
            f = glyph_feature(img) if img is not None else None
            if f:
                self.samples.append((RANKS[name], f, path))

    def missing_ranks(self) -> list[int]:
        return sorted(set(range(1, 14)) - {r for r, _, _ in self.samples})

    def _dists(self, f, skip: int | None = None) -> dict[int, float]:
        best: dict[int, float] = {}
        for i, (r, s, _) in enumerate(self.samples):
            if i != skip:
                d = glyph_distance(f, s)
                if d < best.get(r, 9.0):
                    best[r] = d
        return best

    def _decide(self, best: dict[int, float], strict: bool) -> Read:
        if not best:
            return Read(None, None, 9.0, 1.0)
        order = sorted(best.items(), key=lambda kv: kv[1])
        r, d = order[0]
        ratio = d / max(order[1][1] if len(order) > 1 else 9.0, 1e-6)
        k = 0.5 if strict else 1.0
        ok = d <= self.max_dist * k and ratio <= self.max_ratio * (0.8 if strict else 1.0)
        return Read(r if ok else None, r, d, ratio)

    def read(self, crop, strict: bool = False) -> Read:
        """strict=True (used while some ranks have no samples yet) only accepts very close matches."""
        f = glyph_feature(crop)
        return self._decide(self._dists(f), strict) if f else Read(None, None, 9.0, 1.0)

    def add(self, rank: int, crop) -> str:
        n = 0
        while os.path.exists(os.path.join(self.folder, f"{NAMES[rank]}_{n}.png")):
            n += 1
        path = os.path.join(self.folder, f"{NAMES[rank]}_{n}.png")
        cv2.imwrite(path, crop)
        f = glyph_feature(crop)
        if f:
            self.samples.append((rank, f, path))
        return path

    def leave_one_out(self):
        """For each sample, how it would be read by the other samples: [(path, true_rank, Read)]."""
        return [(p, r, self._decide(self._dists(f, skip=i), False)) for i, (r, f, p) in enumerate(self.samples)]


# ---------- the whole game screen ----------
@dataclass
class Card:
    rank: int | None
    x: int
    y: int
    box: tuple
    crop: np.ndarray = field(repr=False)
    read: Read | None = None
    unlock: float = 0.0


@dataclass
class GameView:
    active: Card | None
    board: list[Card]          # readable, fully visible face-up cards
    unreadable: list[Card]     # face-up cards whose rank isn't certain
    blue_area: int             # face-down card pixels left on the board
    end_level: bool            # red END LEVEL button visible: the draw pile is empty
    pile: np.ndarray = field(repr=False)  # crop of the draw pile's top card
    pile_wild: bool = False    # a free WILD card (streak reward) sits on top of the draw pile
    active_gold: bool = False  # the bottom card slot is golden, i.e. a WILD card is the bottom card
    free_extra: bool = False   # pile empty and the "+N cards" tile is free (no coin price, or a red badge)
    extra_reason: str = ""


def read_game(img, book: GlyphBook, strict: bool = False) -> GameView:
    blue = blue_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    pad = 22
    board, bad = [], []
    for cx, cy, (x, y, w, h), crop in find_cards(img, G.BOARD_ROI):
        rd = book.read(crop, strict)
        ring = blue[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        outside = (ring > 0).sum() - (blue[y:y + h, x:x + w] > 0).sum()
        card = Card(rd.rank, cx, cy, (x, y, w, h), crop, rd, float(min(outside / (0.6 * w * h), 2.0)))
        (board if rd.rank else bad).append(card)

    active = None
    found = find_cards(img, G.ACTIVE_ROI)
    if found:
        cx, cy, box, crop = found[0]
        rd = book.read(crop, strict)
        active = Card(rd.rank, cx, cy, box, crop, rd)

    bx, by, bw, bh = G.BOARD_ROI
    px, py, pw, ph = G.PILE_TOP_ROI
    end_level = fraction(red_mask, img, G.END_LEVEL_ROI) > 0.35
    return GameView(active, board, bad, int((blue[by:by + bh, bx:bx + bw] > 0).sum()), end_level,
                    img[py:py + ph, px:px + pw].copy(),
                    pile_wild=not end_level and fraction(gold_mask, img, G.PILE_TOP_ROI) > 0.45,
                    active_gold=active is None and fraction(gold_mask, img, G.ACTIVE_INNER) > 0.5,
                    **_extra_cards(img, end_level))


def _extra_cards(img, end_level) -> dict:
    """Decide whether the "+N cards" tile can be tapped for free. A shown price always wins."""
    if not end_level:
        return {}
    badge = fraction(red_mask, img, G.PLUS5_BADGE_ROI) > 0.3
    price, score = price_under_tile(img)
    if price is True:
        return {"free_extra": False, "extra_reason": f"coin price shown (match {score:.2f}), never tapping"}
    if price is False:
        return {"free_extra": True, "extra_reason": f"no coin price on the tile (match {score:.2f})"}
    if badge:
        return {"free_extra": True, "extra_reason": "red badge (price check unavailable)"}
    return {"free_extra": False, "extra_reason": "can't verify it is free"}


def wild_card(img) -> "Card":
    """Pseudo card for a WILD lying in the bottom-card slot."""
    x, y, w, h = G.ACTIVE_INNER
    return Card(0, G.ACTIVE_TAP[0], G.ACTIVE_TAP[1], G.ACTIVE_INNER, img[y:y + h, x:x + w].copy())


def find_coin(img, roi):
    """Bounding box (absolute x, y, w, h) of a round coin icon inside roi, or None."""
    rx, ry, rw, rh = roi
    m = coin_mask(cv2.cvtColor(img[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2HSV))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if 150 <= area <= 1500 and 0.6 <= w / h <= 1.6 and area >= 0.55 * w * h and (best is None or area > best[4]):
            best = (x, y, w, h, area)
    return None if best is None else (rx + best[0], ry + best[1], best[2], best[3])


def price_under_tile(img):
    """Is a coin price shown on the "+N cards" tile?  Returns (True / False / None, match score).
    None means it can't be checked.

    The coin under the WILD card (always shown) is cut out as a template and searched for in the
    tile's price area, so this adapts to the game's own coin art without guessing colours.
    """
    ref = find_coin(img, G.WILD_PRICE_ROI)
    if ref is None:
        return None, 0.0
    x, y, w, h = ref
    p = 2
    tpl = cv2.cvtColor(img[y - p:y + h + p, x - p:x + w + p], cv2.COLOR_BGR2GRAY)
    tx, ty, tw, th = G.TILE_PRICE_ROI
    area = cv2.cvtColor(img[ty:ty + th, tx:tx + tw], cv2.COLOR_BGR2GRAY)
    if tpl.size == 0:
        return None, 0.0
    score = 0.0
    for s in (0.7, 0.85, 1.0, 1.15, 1.3):          # the tile's coin may be drawn at another size
        t = cv2.resize(tpl, None, fx=s, fy=s)
        if t.shape[0] <= area.shape[0] and t.shape[1] <= area.shape[1]:
            score = max(score, float(cv2.matchTemplate(area, t, cv2.TM_CCOEFF_NORMED).max()))
    return (score >= 0.6 or find_coin(img, G.TILE_PRICE_ROI) is not None), score


def _small_gray(a):
    return cv2.cvtColor(cv2.resize(a, (24, 40), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)


def image_diff(a, b) -> float:
    """Mean grey-level difference of two crops (0 = identical)."""
    return float(np.abs(_small_gray(a) - _small_gray(b)).mean())


def pile_matches(ref, cur) -> bool:
    """Does the draw pile still look like it did at the start of the level?"""
    return image_diff(ref, cur) < 20
