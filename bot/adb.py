"""adb wrapper. Works in the canonical 1536-px-wide frame so any phone resolution is fine."""
import subprocess
import time

import cv2
import numpy as np

from .geometry import CANON_W


class Phone:
    def __init__(self, serial: str | None = None):
        self.base = ["adb"] + (["-s", serial] if serial else [])
        self.scale = 1.0

    def _run(self, *args, **kw):
        return subprocess.run(self.base + list(args), check=True, capture_output=True, timeout=20, **kw)

    def grab(self) -> np.ndarray:
        """Screenshot resized to the canonical width (BGR)."""
        png = self._run("exec-out", "screencap", "-p").stdout
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Could not decode screenshot. Is the phone connected, unlocked and authorised?")
        self.scale = CANON_W / img.shape[1]
        return cv2.resize(img, (CANON_W, round(img.shape[0] * self.scale)), interpolation=cv2.INTER_AREA)

    def stable_grab(self, tries: int = 8, gap: float = 0.35) -> np.ndarray:
        """Wait until two consecutive frames match, so card animations have finished."""
        prev = self.grab()
        for _ in range(tries):
            time.sleep(gap)
            cur = self.grab()
            a = cv2.cvtColor(cv2.resize(prev, (192, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
            b = cv2.cvtColor(cv2.resize(cur, (192, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
            if np.abs(a - b).mean() < 1.5:
                return cur
            prev = cur
        return prev

    def tap(self, x: float, y: float) -> None:
        """Tap at canonical coordinates."""
        self._run("shell", "input", "tap", str(round(x / self.scale)), str(round(y / self.scale)))

    def keep_awake(self) -> None:
        """Keep the screen on while plugged in over USB."""
        self._run("shell", "svc", "power", "stayon", "usb")
