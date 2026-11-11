import argparse, glob, os
import cv2
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--board", required=True)          # ví dụ 08
ap.add_argument("--x", type=int, required=True)    # tâm điểm báo giả
ap.add_argument("--y", type=int, required=True)
ap.add_argument("--n", type=int, default=6)        # số ảnh ghép
ap.add_argument("--root", default="C:/Project")    # nơi chứa ảnh
ap.add_argument("--csv", default="results/pcb_v8s_board_test_errors/false_pos_list.csv")
a = ap.parse_args()

d = pd.read_csv(a.csv)
d["board"] = d.image.str.extract(r"^(\d+)_")[0]
cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
s = d[(d.board == a.board) & ((cx - a.x).abs() < 60) & ((cy - a.y).abs() < 60)].head(a.n)

paths = {}
for ext in ("jpg", "jpeg", "png", "bmp"):
    for p in glob.glob(os.path.join(a.root, "**", f"*.{ext}"), recursive=True):
        paths[os.path.basename(p)] = p

tiles, half = [], 120
for r in s.itertuples():
    if r.image not in paths:
        print("Không thấy ảnh:", r.image); continue
    im = cv2.imread(paths[r.image])
    ox, oy = max(0, a.x - half), max(0, a.y - half)
    crop = im[oy:a.y + half, ox:a.x + half].copy()
    cv2.rectangle(crop, (int(r.x1) - ox, int(r.y1) - oy), (int(r.x2) - ox, int(r.y2) - oy), (0, 0, 255), 2)
    crop = cv2.resize(crop, (320, 320))
    cv2.putText(crop, r.image[:24], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(crop)

if tiles:
    out = f"results/hotspot_{a.board}_{a.x}_{a.y}.png"
    cv2.imwrite(out, cv2.hconcat(tiles))
    print("Đã lưu", out, "-", len(tiles), "ảnh")