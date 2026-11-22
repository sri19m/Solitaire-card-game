"""One-time labelling of card glyphs. Shows the card and asks you for its rank."""
import cv2

from .vision import RANKS


def ask_rank(crop):
    cv2.imshow("What rank is this card?", cv2.resize(crop, None, fx=2, fy=2))
    cv2.waitKey(50)
    while True:
        s = input("Rank of the card in the window (A,2..10,J,Q,K, or s to skip): ").strip().upper()
        if s == "S":
            cv2.destroyAllWindows()
            return None
        if s in RANKS:
            cv2.destroyAllWindows()
            return RANKS[s]
        print("  not a rank")
