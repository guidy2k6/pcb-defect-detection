"""Chia lai thu muc YOLO co san (chi co train/val, chia ngau nhien) theo bo mach, dung configs/board_split.json.

Khong can PCB_DATASET goc: ID bo lay tu ten file (<ID bo>_...). Anh va nhan duoc COPY, thu muc nguon giu nguyen.
    python data/io/resplit_by_board.py --source C:/Users/Admin/Downloads/pcb_yolo
Ket qua: dataset/pcb_boards_yolo/{images,labels}/{train,val,test}  (configs/data_board.yaml tro toi day)
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

from check_xml import load_assignment

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="thu muc co images/{train,val}, labels/{train,val}")
    ap.add_argument("--output", type=Path, default=Path("dataset/pcb_boards_yolo"))
    ap.add_argument("--manifest", type=Path, default=Path("configs/board_split.json"))
    a = ap.parse_args()

    assignment, dup = load_assignment(a.manifest)
    if dup:
        raise ValueError(f"Boards occur in multiple manifest splits: {dict(dup)}")
    imgs = sorted(p for p in (a.source / "images").rglob("*") if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"Khong co anh trong {a.source / 'images'}")

    seen, counts, boxes = set(), Counter(), Counter()
    for img in imgs:
        if img.name in seen:
            raise SystemExit(f"Trung ten anh o hai tap cu: {img.name}")
        seen.add(img.name)
        bid = img.stem.split("_", 1)[0]
        if bid not in assignment:
            raise SystemExit(f"Bo {bid} (anh {img.name}) khong co trong {a.manifest}")
        split = assignment[bid]
        lbl = a.source / "labels" / img.relative_to(a.source / "images").with_suffix(".txt")
        if not lbl.exists():
            raise SystemExit(f"Thieu nhan: {lbl}")
        for kind, src in (("images", img), ("labels", lbl)):
            dst = a.output / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
        counts[split] += 1
        boxes[split] += sum(1 for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip())

    print(f"{'tap':<7}{'anh':>6}{'box':>7}")
    for s in ("train", "val", "test"):
        print(f"{s:<7}{counts[s]:>6}{boxes[s]:>7}")
    print(f"Tong: {sum(counts.values())} anh, {sum(boxes.values())} box. Ket qua: {a.output.resolve()}")


if __name__ == "__main__":
    main()
