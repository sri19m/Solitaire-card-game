"""Labelling card glyphs by hand. Shows the card in a window and asks for its rank."""
import cv2

from .vision import RANKS

WIN = "Card"
ALIASES = {"1": "A", "T": "10", "0": "10"}


def ask_rank(crop, question: str = "What rank is this card?", allow_keep: bool = False):
    """Return a rank 1-13, None to skip, or "keep" (Enter) when allow_keep is set."""
    cv2.imshow(WIN, cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST))
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.waitKey(50)
    hint = "A,2..10,J,Q,K" + (", Enter = bot was right" if allow_keep else "") + ", s = skip"
    try:
        while True:
            s = input(f"{question} [{hint}]: ").strip().upper()
            s = ALIASES.get(s, s)
            if allow_keep and s == "":
                return "keep"
            if s == "S":
                return None
            if s in RANKS:
                return RANKS[s]
            print("  not a rank")
    finally:
        cv2.destroyWindow(WIN)
        cv2.waitKey(1)
