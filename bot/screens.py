"""Screens you taught the bot (reward popups, "tap to collect", ...).

In --learn mode an unknown screen opens in a window: click where the bot should tap, then
press Enter. Next time a matching screen appears the bot taps there by itself.
Saved as screens/<id>.png (the screenshot) + screens/<id>.json (what to do).
"""
import glob
import json
import os
import time

import cv2
import numpy as np


def _thumb(img, size=(64, 30)):
    return cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)


def _patch(img, pos, half=(120, 45)):
    x, y = pos
    h, w = img.shape[:2]
    x0, y0 = max(x - half[0], 0), max(y - half[1], 0)
    crop = img[y0:min(y + half[1], h), x0:min(x + half[0], w)]
    return _thumb(crop, (60, 22))


class ScreenBook:
    def __init__(self, folder: str = "screens", full_tol: float = 30.0, patch_tol: float = 18.0):
        self.folder, self.full_tol, self.patch_tol = folder, full_tol, patch_tol
        os.makedirs(folder, exist_ok=True)
        self.items = []
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            meta = json.load(open(path))
            img = cv2.imread(path[:-5] + ".png")
            if img is not None:
                self._add(img, meta)

    def _add(self, img, meta):
        pos = tuple(meta["pos"]) if meta.get("pos") else (img.shape[1] // 2, img.shape[0] // 2)
        self.items.append((meta, _thumb(img), _patch(img, pos), pos))

    def match(self, img):
        """Return (meta, pos) of the best matching taught screen, or None."""
        t = _thumb(img)
        best = None
        for meta, thumb, patch, pos in self.items:
            full = float(np.abs(t - thumb).mean())
            local = float(np.abs(_patch(img, pos) - patch).mean())
            if full < self.full_tol and local < self.patch_tol and (best is None or full + local < best[0]):
                best = (full + local, meta, pos)
        return (best[1], best[2]) if best else None

    def remember(self, img, action: str, pos=None, note: str = "") -> dict:
        sid = time.strftime("%Y%m%d_%H%M%S")
        meta = {"action": action, "pos": list(pos) if pos else None, "note": note}
        cv2.imwrite(os.path.join(self.folder, sid + ".png"), img)
        json.dump(meta, open(os.path.join(self.folder, sid + ".json"), "w"))
        self._add(img, meta)
        return meta

    def teach(self, img):
        """Ask the user what to do on this screen. Returns the saved meta, or None if skipped."""
        win = "Unknown screen: click where to tap"
        scale = 0.8
        state = {"pos": None}

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["pos"] = (int(x / scale), int(y / scale))

        print("\nUnknown screen. In the window: click where the bot should tap, then press Enter.\n"
              "  w = this screen goes away by itself (wait)   s = stop the bot here   Esc = skip\n"
              "  Never teach a tap on anything that costs coins.")
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        try:
            while True:
                show = cv2.resize(img, None, fx=scale, fy=scale)
                if state["pos"]:
                    p = (int(state["pos"][0] * scale), int(state["pos"][1] * scale))
                    cv2.circle(show, p, 18, (0, 0, 255), 3)
                cv2.imshow(win, show)
                k = cv2.waitKey(30) & 0xFF
                if k in (13, 10) and state["pos"]:
                    return self.remember(img, "tap", state["pos"])
                if k == ord("w"):
                    return self.remember(img, "wait")
                if k == ord("s"):
                    return self.remember(img, "stop")
                if k == 27:
                    return None
        finally:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
