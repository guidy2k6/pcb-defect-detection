"""Cat vung bao gia tu DUNG anh test ma analyze_errors.py da dung.

Dat file nay tai tools/crop_hotspot_v2.py, chay tu thu muc goc repo:
  python tools\\crop_hotspot_v2.py --board 08 --x 180 --y 1080
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_utils import load_split  # cung ham analyze_errors.py dung de tim anh

ap = argparse.ArgumentParser()
ap.add_argument("--board", required=True)            # vi du 08
ap.add_argument("--x", type=int, required=True)      # tam diem bao gia
ap.add_argument("--y", type=int, required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--data", default="configs/data_board.yaml")
ap.add_argument("--csv", default="results/pcb_v8s_board_test_errors/false_pos_list.csv")
a = ap.parse_args()

d = pd.read_csv(a.csv)
d["board"] = d.image.str.extract(r"^(\d+)_")[0]
cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
s = d[(d.board == a.board) & ((cx - a.x).abs() < 60) & ((cy - a.y).abs() < 60)].head(a.n)

_, images = load_split(a.data, "test")
paths = {Path(p).name: str(p) for p in images}
print("So anh test:", len(paths), "| thu muc:", Path(next(iter(paths.values()))).parent)

half, raws, tiles = 120, [], []
for r in s.itertuples():
    if r.image not in paths:
        print("Khong thay anh trong split test:", r.image)
        continue
    im = cv2.imread(paths[r.image])
    h, w = im.shape[:2]
    ox, oy = max(0, a.x - half), max(0, a.y - half)
    raw = im[oy:a.y + half, ox:a.x + half].copy()
    raws.append(cv2.resize(raw, (240, 240)).astype(np.float32))
    tile = raw.copy()
    cv2.rectangle(tile, (int(r.x1) - ox, int(r.y1) - oy), (int(r.x2) - ox, int(r.y2) - oy), (0, 0, 255), 2)
    tile = cv2.resize(tile, (320, 320))
    cv2.putText(tile, f"{r.image[:22]} {w}x{h}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    tiles.append(tile)

for i in range(1, len(raws)):
    print(f"Do lech pixel TB giua anh 1 va anh {i + 1}: {np.abs(raws[0] - raws[i]).mean():.1f} / 255")

if tiles:
    out = f"results/hotspot2_{a.board}_{a.x}_{a.y}.png"
    cv2.imwrite(out, cv2.hconcat(tiles))
    print("Da luu", out)