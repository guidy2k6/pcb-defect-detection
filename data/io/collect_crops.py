"""Cat vung loi theo nhan thanh anh nho cho bo phan loai (khoi Dataset, Collect Data cua Hinh 1.1).

Doc thang dataset/pcb_boards_yolo (da chia theo bo mach boi divide_data.py hoac resplit_by_board.py),
nen cach chia giong het YOLO: cung mot bo khong xuat hien o hai tap.
Ket qua: dataset/cls/{train,val,test}/<lop>/<ten_anh>_<k>.png
    python data/io/collect_crops.py
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from crop_utils import CROP_MARGIN, CROP_SIZE, class_folder, crop_square  # noqa: E402
from yolo_utils import load_split, read_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=REPO / "configs" / "data_board.yaml")
    ap.add_argument("--output", type=Path, default=REPO / "dataset" / "cls")
    ap.add_argument("--margin", type=float, default=CROP_MARGIN, help="le them quanh box (ti le canh)")
    ap.add_argument("--size", type=int, default=CROP_SIZE)
    a = ap.parse_args()

    stats = {}
    for split in ("train", "val", "test"):
        names, images = load_split(a.data, split)
        counts = {i: 0 for i in names}
        n_labels = 0
        for img_path in images:
            labels = read_labels(img_path)
            n_labels += len(labels)
            if not labels:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(img_path)
            h, w = img.shape[:2]
            for k, (cid, xc, yc, bw, bh) in enumerate(labels):
                x1, x2 = (xc - bw / 2) * w, (xc + bw / 2) * w
                y1, y2 = (yc - bh / 2) * h, (yc + bh / 2) * h
                folder = a.output / split / class_folder(names[cid])
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{img_path.stem}_{k}.png"), crop_square(img, x1, y1, x2, y2, a.margin, a.size))
                counts[cid] += 1
        if sum(counts.values()) != n_labels:
            raise SystemExit(f"Tap {split}: cat {sum(counts.values())} anh nhung co {n_labels} nhan")
        stats[split] = (names, counts)

    names = stats["train"][0]
    print(f"{'tap':<7}" + "".join(f"{names[i]:>17}" for i in names) + f"{'tong':>8}")
    for split, (_, counts) in stats.items():
        print(f"{split:<7}" + "".join(f"{counts[i]:>17}" for i in names) + f"{sum(counts.values()):>8}")
    print(f"Ket qua: {a.output.resolve()}")


if __name__ == "__main__":
    main()
