import argparse
import csv
from pathlib import Path

import numpy as np

from yolo_utils import load_split, read_labels

REPO = Path(__file__).resolve().parents[1]


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(gt_xyxy, gt_cls, pr_xyxy, pr_cls, pr_conf, thr):
    """Ghep 1-1. Tra ve (pairs[(gi, pi)], gt_missed[list gi], pred_extra[list pi])."""
    iou = iou_matrix(pr_xyxy, gt_xyxy)  # (P, G)
    used_g, pairs = set(), []
    for pi in np.argsort(-np.asarray(pr_conf), kind="stable"):
        if iou.shape[1] == 0:
            break
        cand = [(iou[pi, gi], gi) for gi in range(iou.shape[1]) if gi not in used_g and iou[pi, gi] >= thr]
        if cand:
            _, gi = max(cand)
            used_g.add(gi)
            pairs.append((gi, int(pi)))
    matched_p = {p for _, p in pairs}
    missed = [g for g in range(len(gt_xyxy)) if g not in used_g]
    extra = [p for p in range(len(pr_xyxy)) if p not in matched_p]
    return pairs, missed, extra


def summarize(records, n_cls):
    """records: list of dict(gt_cls, gt_area, status, pred_cls) cho GT, va list pred_extra_cls.
    Tra ve (ma tran (n_cls+1)x(n_cls+1) hang=lop that|nen, cot=lop du doan|bo sot)."""
    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=int)  # chi so n_cls = nen / bo sot
    for r in records["gt"]:
        if r["status"] == "missed":
            cm[r["cls"], n_cls] += 1
        else:
            cm[r["cls"], r["pred_cls"]] += 1
    for pc in records["extra"]:
        cm[n_cls, pc] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO  # import tre de test ham thuan khong can cai ultralytics

    names, images = load_split(a.data, a.split)
    n = len(names)
    model = YOLO(a.weights)
    rec = {"gt": [], "extra": []}
    for r in model.predict(source=[str(p) for p in images], stream=True, verbose=False,
                           imgsz=a.imgsz, conf=a.conf, device=a.device):
        h, w = r.orig_shape
        img = Path(r.path)
        gts = read_labels(img)
        g_cls = np.array([g[0] for g in gts], dtype=int)
        g_xyxy = np.array([[(g[1] - g[3] / 2) * w, (g[2] - g[4] / 2) * h,
                            (g[1] + g[3] / 2) * w, (g[2] + g[4] / 2) * h] for g in gts]).reshape(-1, 4)
        p_xyxy = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        p_cls = r.boxes.cls.cpu().numpy().astype(int)
        p_conf = r.boxes.conf.cpu().numpy()

        pairs, missed, extra = match_image(g_xyxy, g_cls, p_xyxy, p_cls, p_conf, a.match_iou)
        area = (g_xyxy[:, 2] - g_xyxy[:, 0]) * (g_xyxy[:, 3] - g_xyxy[:, 1]) / (w * h) if len(gts) else []
        for gi, pi in pairs:
            ok = g_cls[gi] == p_cls[pi]
            rec["gt"].append({"cls": int(g_cls[gi]), "area": float(area[gi]),
                              "status": "ok" if ok else "confused", "pred_cls": int(p_cls[pi])})
        for gi in missed:
            rec["gt"].append({"cls": int(g_cls[gi]), "area": float(area[gi]), "status": "missed", "pred_cls": -1})
        rec["extra"] += [int(p_cls[pi]) for pi in extra]

    n_gt = len(rec["gt"])
    n_ok = sum(r["status"] == "ok" for r in rec["gt"])
    n_conf = sum(r["status"] == "confused" for r in rec["gt"])
    n_miss = sum(r["status"] == "missed" for r in rec["gt"])
    n_fp = len(rec["extra"])
    assert n_gt == n_ok + n_conf + n_miss

    cm = summarize(rec, n)
    out = REPO / "results" / f"{a.name}_{a.split}_errors"
    out.mkdir(parents=True, exist_ok=True)
    labels = [names[i] for i in range(n)]

    with open(out / "confusion_true_x_pred.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels + ["missed"])
        for i in range(n):
            w.writerow([labels[i]] + [int(x) for x in cm[i, :n]] + [int(cm[i, n])])
        w.writerow(["background(false_pos)"] + [int(x) for x in cm[n, :n]] + [""])

    with open(out / "error_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "gt", "correct", "confused", "missed", "false_pos_by_pred_class"])
        for i in range(n):
            gi = [r for r in rec["gt"] if r["cls"] == i]
            w.writerow([labels[i], len(gi), sum(r["status"] == "ok" for r in gi),
                        sum(r["status"] == "confused" for r in gi),
                        sum(r["status"] == "missed" for r in gi), int(cm[n, i])])

    areas = np.array([r["area"] for r in rec["gt"]])
    q1, q2 = np.percentile(areas, [33.3, 66.7]) if len(areas) else (0, 0)
    groups = [("nho", areas <= q1), ("vua", (areas > q1) & (areas <= q2)), ("lon", areas > q2)]
    st = np.array([r["status"] for r in rec["gt"]])
    with open(out / "size_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "area_max_frac", "gt", "missed", "miss_rate", "confused", "confused_rate"])
        for (gname, mask), amax in zip(groups, (q1, q2, areas.max() if len(areas) else 0)):
            k = int(mask.sum())
            mi, co = int((st[mask] == "missed").sum()), int((st[mask] == "confused").sum())
            w.writerow([gname, f"{amax:.6f}", k, mi, f"{mi / k if k else 0:.4f}", co, f"{co / k if k else 0:.4f}"])

    print(f"Tham so co dinh: conf={a.conf}, match IoU={a.match_iou}, imgsz={a.imgsz}, tap={a.split}")
    print(f"Nhan: {n_gt} | dung {n_ok} | nham lop {n_conf} | bo sot {n_miss} | bao thua {n_fp}")
    print(f"Kiem tra: nhan = {n_ok}+{n_conf}+{n_miss} = {n_ok + n_conf + n_miss}; "
          f"du doan = {n_ok}+{n_conf}+{n_fp} = {n_ok + n_conf + n_fp}")
    print(f"Ket qua: {out}")


if __name__ == "__main__":
    main()