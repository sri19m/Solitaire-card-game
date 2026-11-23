"""adb wrapper. Works in the canonical 1536-px-wide frame so any phone resolution is fine.

Speed: screenshots are fetched raw (no PNG compression on the phone, same pixels), taps go
through one persistent `adb shell`, and "wait until the animation stopped" checks often.
"""
import queue
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

from .geometry import CANON_W


class Phone:
    def __init__(self, serial: str | None = None, raw: bool = True):
        self.base = ["adb"] + (["-s", serial] if serial else [])
        self.scale = 1.0
        self.raw = raw
        self.shell = None
        self.pending = False

    def _run(self, *args, **kw):
        return subprocess.run(self.base + list(args), check=True, capture_output=True, timeout=20, **kw)

    # ---------- screenshots ----------
    def _grab_raw(self):
        data = self._run("exec-out", "screencap").stdout
        if len(data) < 16:
            return None
        w, h, fmt = np.frombuffer(data[:12], np.uint32)
        header = len(data) - int(w) * int(h) * 4
        if fmt not in (1, 2) or header not in (12, 16):   # only RGBA/RGBX_8888 handled here
            return None
        rgba = np.frombuffer(data, np.uint8, offset=header).reshape(int(h), int(w), 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def _grab_png(self):
        png = self._run("exec-out", "screencap", "-p").stdout
        return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)

    def grab(self) -> np.ndarray:
        """Screenshot resized to the canonical width (BGR)."""
        self.sync()
        img = None
        if self.raw:
            img = self._grab_raw()
            if img is None:
                self.raw = False          # this phone doesn't give raw RGBA: use PNG from now on
        if img is None:
            img = self._grab_png()
        if img is None:
            raise RuntimeError("Could not read a screenshot. Is the phone connected, unlocked and authorised?")
        self.scale = CANON_W / img.shape[1]
        return cv2.resize(img, (CANON_W, round(img.shape[0] * self.scale)), interpolation=cv2.INTER_AREA)

    def stable_grab(self, min_wait: float = 0.0, gap: float = 0.12, max_wait: float = 3.0) -> np.ndarray:
        """Wait until two consecutive frames match, so card animations have finished."""
        t0 = time.time()
        if min_wait:
            time.sleep(min_wait)
        prev = self.grab()
        while time.time() - t0 < max_wait:
            time.sleep(gap)
            cur = self.grab()
            a = cv2.cvtColor(cv2.resize(prev, (192, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
            b = cv2.cvtColor(cv2.resize(cur, (192, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
            if np.abs(a - b).mean() < 1.5:
                return cur
            prev = cur
        return prev

    # ---------- taps ----------
    def _open_shell(self):
        # Binary pipes: Windows text mode would send "\r\n" and break the tap arguments.
        self.shell = subprocess.Popen(self.base + ["shell"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, bufsize=0)
        self.lines = queue.Queue()
        out = self.shell.stdout

        def reader():
            for line in iter(out.readline, b""):
                self.lines.put(line)

        threading.Thread(target=reader, daemon=True).start()

    def tap(self, x: float, y: float) -> None:
        """Tap at canonical coordinates (queued on the persistent shell)."""
        cmd = f"input tap {round(x / self.scale)} {round(y / self.scale)}\n"
        for attempt in range(2):
            try:
                if self.shell is None or self.shell.poll() is not None:
                    self._open_shell()
                self.shell.stdin.write(cmd.encode())
                self.shell.stdin.flush()
                self.pending = True
                return
            except (BrokenPipeError, OSError):
                self.shell = None
        self._run("shell", "input", "tap", str(round(x / self.scale)), str(round(y / self.scale)))

    def sync(self, timeout: float = 10.0) -> None:
        """Block until every queued tap has been executed on the phone."""
        if not self.pending or self.shell is None:
            return
        marker = f"done-{uuid.uuid4().hex[:8]}"
        try:
            self.shell.stdin.write(f"echo {marker}\n".encode())
            self.shell.stdin.flush()
            end = time.time() + timeout
            while time.time() < end:
                try:
                    if marker.encode() in self.lines.get(timeout=max(end - time.time(), 0.01)):
                        break
                except queue.Empty:
                    break
        except (BrokenPipeError, OSError):
            self.shell = None
        self.pending = False

    def keep_awake(self) -> None:
        """Keep the screen on while plugged in over USB."""
        self._run("shell", "svc", "power", "stayon", "usb")
