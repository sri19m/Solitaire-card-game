"""Check your labelled cards: counts per rank, cards the others would misread, and a picture of all labels.

    python tools/check_glyphs.py      -> prints a report and writes glyph_sheet.png
Delete a wrong file from glyphs/ to remove a bad label.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.vision import FACE_H, FACE_W, NAMES, GlyphBook

cfg = json.load(open("config.json"))
book = GlyphBook(cfg["glyph_dir"], cfg["max_glyph_dist"], cfg["max_glyph_ratio"])
counts = Counter(r for r, _, _ in book.samples)
print("samples per rank:", "  ".join(f"{NAMES[r]}:{counts.get(r, 0)}" for r in range(1, 14)))
if book.missing_ranks():
    print("MISSING:", ", ".join(NAMES[r] for r in book.missing_ranks()))

wrong = unsure = 0
for path, r, rd in book.leave_one_out():
    if counts[r] < 2:
        continue  # nothing to compare against yet
    if rd.guess != r:
        wrong += 1
        print(f"  {path}: labelled {NAMES[r]} but looks like {NAMES.get(rd.guess, '?')} (check the label)")
    elif rd.rank is None:
        unsure += 1
print(f"leave-one-out: {wrong} would be misread, {unsure} would be asked about")

rows = []
for r in range(1, 14):
    imgs = [cv2.resize(cv2.imread(p), (FACE_W, FACE_H)) for rr, _, p in book.samples if rr == r][:12]
    row = np.full((FACE_H, FACE_W * 13, 3), 60, np.uint8)
    for i, im in enumerate(imgs):
        row[:, (i + 1) * FACE_W:(i + 2) * FACE_W] = im
    cv2.putText(row, NAMES[r], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    rows.append(row)
cv2.imwrite("glyph_sheet.png", np.vstack(rows))
print("wrote glyph_sheet.png (one row per rank; every card in a row should show that rank)")
