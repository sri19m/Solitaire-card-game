"""Measure how fast your phone delivers screenshots, and check raw and PNG give the same picture.

    python tools/bench_adb.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.adb import Phone

p = Phone()
for name, fn in (("PNG", p._grab_png), ("raw", p._grab_raw)):
    t = time.time()
    imgs = [fn() for _ in range(5)]
    dt = (time.time() - t) / 5
    ok = all(i is not None for i in imgs)
    print(f"{name:4s} screenshot: {dt * 1000:6.0f} ms each" + ("" if ok else "  (not supported on this phone)"))
a, b = p._grab_png(), p._grab_raw()
if a is not None and b is not None and a.shape == b.shape:
    print(f"raw vs PNG pixel difference: {np.abs(a.astype(int) - b.astype(int)).mean():.2f} (0 = identical)")
