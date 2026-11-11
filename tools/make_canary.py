"""Create a tiny set of labelled reference crops (the "canary") that the demo uses to check the crop classifier at start-up.

    python tools/make_canary.py --data configs/data_board.yaml --split val --per-class 2
Writes demo/assets/canary/<class>_<k>.png (128x128 crops of ground-truth boxes, same recipe as the classifier's training crops).
The demo classifies them on start and warns if most are wrong, which is what an unpatched (class-name-permuted) best_cls.pt does.
Use val or train crops; they are only a plumbing check, not a measurement.
"""
import argparse
from pathlib import Path

import cv2

from crop_utils import class_folder, crop_square
from yolo_utils import load_split, read_labels

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "configs" / "data_board.yaml"))
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", type=Path, default=REPO / "demo" / "assets" / "canary")
    a = ap.parse_args()

    names, images = load_split(a.data, a.split)
    a.out.mkdir(parents=True, exist_ok=True)
    taken = {c: 0 for c in names}
    for path in images:
        if all(n >= a.per_class for n in taken.values()):
            break
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for c, xc, yc, bw, bh in read_labels(path):
            c = int(c)
            if taken[c] >= a.per_class or bw * w < 12 or bh * h < 12:
                continue
            crop = crop_square(img, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)
            taken[c] += 1
            cv2.imwrite(str(a.out / f"{class_folder(names[c])}_{taken[c]}.png"), crop)
    for c, n in taken.items():
        print(f"{names[c]}: {n} crop")
    print(f"Ket qua: {a.out}")


if __name__ == "__main__":
    main()
