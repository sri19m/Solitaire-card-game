"""Annotated screenshots for diagnosing what the bot saw and decided."""
import cv2

from .vision import NAMES


def annotate(img, view, chain, is_suspect, note: str):
    out = img.copy()
    for c in view.board:
        x, y, w, h = c.box
        col = (0, 0, 255) if is_suspect(c) else (0, 200, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        cv2.putText(out, NAMES[c.rank], (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    for c in view.unreadable:
        x, y, w, h = c.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 255), 2)
        g = NAMES.get(c.read.guess, "?") if c.read else "?"
        cv2.putText(out, f"?{g} {c.read.dist:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    if view.active:
        x, y, w, h = view.active.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 120, 0), 2)
    for i, c in enumerate(chain, 1):
        cv2.putText(out, f"#{i}", (c.x - 15, c.y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
    cv2.putText(out, note, (10, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
