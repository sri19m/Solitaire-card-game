"""Label cards the bot saved to pending/ during autonomous play (cards it wasn't sure about).

    python tools/label_pending.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.learn import ask_rank
from bot.vision import GlyphBook

cfg = json.load(open("config.json"))
book = GlyphBook(cfg["glyph_dir"], cfg["max_glyph_dist"], cfg["max_glyph_ratio"])
files = sorted(glob.glob("pending/*.png"))
print(f"{len(files)} cards to label")
for path in files:
    r = ask_rank(cv2.imread(path), f"{os.path.basename(path)}: rank?")
    if r:
        book.add(r, cv2.imread(path))
    os.remove(path)
print("done. Run tools/check_glyphs.py to review all labels.")
