"""Cat anh vi du cho slide phan tich loi: bo sot, nham lop, du doan thua (mac dinh tap val).

    python tools/error_examples.py --weights models/best.pt --data configs/data_board.yaml --split val --per-group 4
Ket qua: results/error_examples/<nhom>/*.jpg, GT khung xanh la, du doan khung do, chu ghi lop that va lop du doan.
Chi dung de minh hoa/giai thich, khong dung de chon lai mo hinh.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from yolo_utils import load_split, read_labels

REPO = Path(__file__).resolve().parents[1]


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt, pr, thr):
    """Ghep tham lam theo IoU giam dan, khong xet lop. Tra ve (cac cap, GT chua ghep, du doan chua ghep)."""
    iou = iou_matrix(gt, pr)
    pairs, ug, up = [], set(), set()
    if iou.size:
        for flat in np.argsort(-iou, axis=None):
            g, p = (int(v) for v in np.unravel_index(flat, iou.shape))
            if iou[g, p] < thr:
                break
            if g in ug or p in up:
                continue
            ug.add(g); up.add(p); pairs.append((g, p))
    return pairs, [g for g in range(len(gt)) if g not in ug], [p for p in range(len(pr)) if p not in up]


def render(img, gt_box, pred_box, text, size=640):
    ref = gt_box if gt_box is not None else pred_box
    h, w = img.shape[:2]
    side = int(min(max(max(ref[2] - ref[0], ref[3] - ref[1]) * 4, 240), h, w))
    cx, cy = (ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2
    x0 = int(min(max(cx - side / 2, 0), w - side)); y0 = int(min(max(cy - side / 2, 0), h - side))
    crop = img[y0:y0 + side, x0:x0 + side].copy()
    for box, color in ((gt_box, (0, 200, 0)), (pred_box, (0, 0, 255))):
        if box is not None:
            cv2.rectangle(crop, (int(box[0] - x0), int(box[1] - y0)), (int(box[2] - x0), int(box[3] - y0)), color, 2)
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(out, (0, 0), (size, 44), (30, 30, 30), -1)
    cv2.putText(out, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(REPO / "configs" / "data_board.yaml"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "error_examples")
    a = ap.parse_args()

    model = YOLO(a.weights)
    names, images = load_split(a.data, a.split)
    items = {"bo_sot": [], "nham_lop": [], "du_doan_thua": []}
    for path in images:
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        labels = read_labels(path)
        gt = np.asarray([[(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
                         for _, xc, yc, bw, bh in labels], np.float32).reshape(-1, 4)
        gc = [c for c, *_ in labels]
        r = model.predict(img, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        pc = r.boxes.cls.cpu().numpy().astype(int)
        pf = r.boxes.conf.cpu().numpy()
        pairs, miss, extra = match(gt, pb, a.match_iou)
        for g, p in pairs:
            if gc[g] != pc[p]:
                items["nham_lop"].append((path, gt[g], pb[p], f"that: {names[gc[g]]} | du doan: {names[pc[p]]} {pf[p]:.2f}"))
        for g in miss:
            items["bo_sot"].append((path, gt[g], None, f"bo sot: {names[gc[g]]}"))
        for p in extra:
            items["du_doan_thua"].append((path, None, pb[p], f"du doan thua: {names[pc[p]]} {pf[p]:.2f}"))

    for group, lst in items.items():
        step = max(1, len(lst) // a.per_group) if lst else 1
        chosen = lst[::step][:a.per_group]  # rai deu qua cac anh, khong chon tay
        (a.out / group).mkdir(parents=True, exist_ok=True)
        for k, (path, g, p, text) in enumerate(chosen):
            cv2.imwrite(str(a.out / group / f"{path.stem}_{k}.jpg"), render(cv2.imread(str(path)), g, p, text))
        print(f"{group}: {len(lst)} truong hop, luu {len(chosen)} anh")
    print(f"Ket qua: {a.out}")


if __name__ == "__main__":
    main()