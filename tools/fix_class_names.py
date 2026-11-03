"""Sua ten lop ghi trong file weights (khong train lai, trong so giu nguyen).

Nguyen nhan: file nhan YOLO danh so lop theo thu tu abc (0 Missing Hole ... 5 Spurious Copper), con configs/data_board.yaml
tung ghi thu tu khac, nen ten lop trong best.pt va trong thu muc dataset/cls bi hoan vi (Short va Spurious Copper dung).

    python tools/fix_class_names.py det runs/detect/pcb_v8s_board/weights/best.pt models/best.pt
    python tools/fix_class_names.py cls runs/classify/pcb_cls_board/weights/best.pt models/best_cls.pt
File nguon khong bi sua. Chay lai tren file da sua se bi tu choi (ten da dung).
"""
import argparse

from ultralytics import YOLO

OLD = {
    "det": {0: "Mouse Bite", 1: "Spur", 2: "Missing Hole", 3: "Short", 4: "Open Circuit", 5: "Spurious Copper"},
    "cls": {0: "missing_hole", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: "spur", 5: "spurious_copper"},
}
NEW = {
    "det": {0: "Missing Hole", 1: "Mouse Bite", 2: "Open Circuit", 3: "Short", 4: "Spur", 5: "Spurious Copper"},
    "cls": {0: "open_circuit", 1: "missing_hole", 2: "spur", 3: "short", 4: "mouse_bite", 5: "spurious_copper"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["det", "cls"])
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    m = YOLO(a.src)
    cur = dict(m.model.names)
    if cur != OLD[a.kind]:
        raise SystemExit(f"Ten lop hien tai khong phai ten cu can sua, khong lam gi: {cur}")
    m.model.names = NEW[a.kind]
    m.save(a.dst)
    chk = YOLO(a.dst).model.names
    assert dict(chk) == NEW[a.kind], chk
    print("Da sua:", {k: (OLD[a.kind][k], NEW[a.kind][k]) for k in OLD[a.kind] if OLD[a.kind][k] != NEW[a.kind][k]})
    print("Ghi:", a.dst)


if __name__ == "__main__":
    main()