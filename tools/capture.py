"""Save a screenshot from the phone into shots/ (use these for templates and tuning)."""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.adb import Phone

name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%H%M%S")
path = Path("shots") / f"{name}.png"
path.parent.mkdir(exist_ok=True)
img = Phone().grab()
cv2.imwrite(str(path), img)
print(f"Saved {path} ({img.shape[1]}x{img.shape[0]})")
