"""Chan doan va danh gia duong XOR (baseline) tren cac bo mach cua tap val hoac test.

    # 1) Chinh tham so tren VAL (khong cham vao test)
    python tools/eval_pipeline.py --dataset PCB_DATASET --split val --overlay 6
    #    Quet luoi close x min_area x grow, in cac to hop co F1 cao nhat o IoU 0,5 (chi cho phep tren val):
    python tools/eval_pipeline.py --dataset PCB_DATASET --split val --sweep
    # 2) Ghi tham so tot nhat vao XOR_PARAMS trong tools/pipeline.py, roi chay TEST mot lan, co ca do chinh xac phan loai tren box tim duoc
    python tools/eval_pipeline.py --dataset PCB_DATASET --split test --cls models/best_cls.pt --iou-cls 0.5

Ket qua: results/pipeline_<split>.csv (P/R theo IoU cho hai bien the: orig = baseline_xor, new = them mask vien + noi manh),
         results/pipeline_<split>_cls.csv (neu co --cls), results/pipeline_<split>_overlay/ (GT xanh la, box do).
"""
import argparse
import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from baseline_xor import IMAGE_EXTENSIONS
from crop_utils import class_folder
from eval_baseline import _align_ground_truth
from pipeline import XOR_PARAMS, boxes_from, classify, classify_boxes, components, draw, xor_detect

REPO = Path(__file__).resolve().parents[1]


def read_gt(xml_path):
    root = ET.parse(xml_path).getroot()
    size = (int(root.findtext("size/width")), int(root.findtext("size/height")))
    boxes, names = [], []
    for o in root.findall("object"):
        b = o.find("bndbox")
        boxes.append([float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        names.append(class_folder(o.findtext("name")))
    return np.asarray(boxes, np.float32).reshape(-1, 4), names, size


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt, pr, thr):
    """Ghep tham lam theo IoU giam dan, moi GT va moi box du doan dung toi da mot lan. Tra ve list (gi, pi)."""
    iou = iou_matrix(gt, pr)
    if iou.size == 0:
        return []
    pairs, used_g, used_p = [], set(), set()
    for flat in np.argsort(-iou, axis=None):
        g, p = np.unravel_index(flat, iou.shape)
        if iou[g, p] < thr:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p); pairs.append((g, p))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="thu muc PCB_DATASET (co images, Annotations, PCB_USED)")
    ap.add_argument("--manifest", type=Path, default=REPO / "configs" / "board_split.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--cls", type=Path, help="best_cls.pt; co thi do them do chinh xac phan loai")
    ap.add_argument("--variant", default="new", choices=["orig", "new"], help="bien the dung cho phan loai")
    ap.add_argument("--iou", type=float, nargs="*", default=[0.5, 0.3, 0.1])
    ap.add_argument("--crop-from", default="original", choices=["original", "aligned"],
                    help="cat vung phan loai tu anh loi goc (khop luc train) hay tu anh da warp")
    ap.add_argument("--iou-cls", type=float, default=0.5, help="IoU toi thieu de coi box la tim dung truoc khi phan loai")
    ap.add_argument("--close", type=int, help=f"kernel dong noi manh XOR (mac dinh {XOR_PARAMS['close']}, trong pipeline.XOR_PARAMS)")
    ap.add_argument("--min-area", type=int, help=f"dien tich toi thieu cua vung XOR (mac dinh {XOR_PARAMS['min_area']})")
    ap.add_argument("--grow", type=float, help=f"he so phong box quanh tam (mac dinh {XOR_PARAMS['grow']})")
    ap.add_argument("--sweep", action="store_true", help="quet luoi tham so tren val (cam tren test)")
    ap.add_argument("--limit", type=int, default=0, help="chi chay N anh dau (thu nhanh)")
    ap.add_argument("--overlay", type=int, default=0, help="luu N anh minh hoa dau tien")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    a = ap.parse_args()

    if a.sweep and a.split != "val":
        raise SystemExit("--sweep chi chay tren val: khong chinh tham so bang tap test.")
    boards = {str(b) for b in json.loads(a.manifest.read_text(encoding="utf-8"))[a.split]}
    templates = {p.stem: p for p in (a.dataset / "PCB_USED").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    xmls = {p.stem.casefold(): p for p in (a.dataset / "Annotations").rglob("*.xml")}
    images = sorted(p for p in (a.dataset / "images").rglob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.split("_", 1)[0] in boards)
    if a.limit:
        images = images[:a.limit]
    if not images:
        raise SystemExit(f"Khong co anh cua cac bo {sorted(boards)} trong {a.dataset / 'images'}")

    cls_model = None
    if a.cls:
        from ultralytics import YOLO
        cls_model = YOLO(str(a.cls))

    stat = {(v, t): [0, 0, 0] for v in ("orig", "new") for t in a.iou}  # tp, n_pred, n_gt
    names = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]  # thu tu thu muc = thu tu lop cua bo phan loai
    conf_mat = np.zeros((len(names), len(names)), int)
    cls_ok = cls_n = gt_ok = gt_n = 0
    overlay_dir = a.out / f"pipeline_{a.split}_overlay"
    t_all = time.perf_counter()
    failed = []
    geo = []
    sweep_data = []  # (gt da can chinh, {close: (stats, shape)})
    CLOSES, AREAS, GROWS = (5, 9, 15, 21), (40, 80, 150, 250), (2.0, 2.3, 2.7, 3.1, 3.5)

    for k, img_path in enumerate(images):
        board = img_path.stem.split("_", 1)[0]
        gt, gt_names, size = read_gt(xmls[img_path.stem.casefold()])
        try:
            r = xor_detect(templates[board], img_path, a.close, a.min_area, a.grow)
        except Exception as e:  # can chinh that bai = bo sot toan bo nhan cua anh
            failed.append((img_path.name, str(e)))
            for key in stat:
                stat[key][2] += len(gt)
            continue
        gt_al = _align_ground_truth(gt, size, r["defect_shape"], r["H"])
        if a.sweep:
            sweep_data.append((gt_al, {c: components(r["xor_raw"], r["valid"], c) for c in CLOSES}))
        for v in ("orig", "new"):
            for t in a.iou:
                s = stat[(v, t)]
                s[0] += len(match(gt_al, r[v], t)); s[1] += len(r[v]); s[2] += len(gt_al)

        for g, p in match(gt_al, r["new"], 0.1):  # hinh hoc cua cac cap chong lan: sai kich thuoc hay lech vi tri?
            gb, pb = gt_al[g], r["new"][p]
            gw, gh = gb[2] - gb[0], gb[3] - gb[1]
            geo.append([(pb[2] - pb[0]) / gw, (pb[3] - pb[1]) / gh,
                        ((pb[0] + pb[2]) - (gb[0] + gb[2])) / 2 / gw, ((pb[1] + pb[3]) - (gb[1] + gb[3])) / 2 / gh])

        if cls_model is not None:
            for (n_, _), g_name in zip(classify(cls_model, r["defect"], gt), gt_names):  # kiem tra: phan loai thang box NHAN tren anh goc
                gt_ok += n_ == g_name; gt_n += 1
            pairs = match(gt_al, r[a.variant], a.iou_cls)
            if pairs:
                preds = classify_boxes(cls_model, r, r[a.variant][[p for _, p in pairs]], a.crop_from)
                for (g, _), (name, _) in zip(pairs, preds):
                    conf_mat[names.index(gt_names[g]), names.index(name)] += 1
                    cls_ok += name == gt_names[g]; cls_n += 1

        if k < a.overlay:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            im = draw(r["aligned"], gt_al, color=(0, 200, 0))
            im = draw(im, r["new"], color=(0, 0, 255))
            cv2.imwrite(str(overlay_dir / f"{img_path.stem}.jpg"), cv2.resize(im, None, fx=0.5, fy=0.5))
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(images)}", flush=True)

    secs = (time.perf_counter() - t_all) / len(images)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\nTap {a.split}: {len(images)} anh, bo {sorted(boards)}, loi can chinh: {len(failed)}")
    print(f"{'bien the':<9}{'IoU':>5}{'GT':>6}{'du doan':>9}{'TP':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for (v, t), (tp, npred, ngt) in stat.items():
        p = tp / npred if npred else 0.0; rc = tp / ngt if ngt else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        if t == a.iou[0]:
            print(f"  Pd (cong thuc 9 cua bai bao, {v}): {abs(npred - ngt) / max(ngt, 1) * 100:.1f}%")
        rows.append([v, t, ngt, npred, tp, f"{p:.4f}", f"{rc:.4f}", f"{f1:.4f}"])
        print(f"{v:<9}{t:>5}{ngt:>6}{npred:>9}{tp:>6}{p:>8.3f}{rc:>8.3f}{f1:>8.3f}")
    with open(a.out / f"pipeline_{a.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["variant", "iou", "gt", "pred", "tp", "P", "R", "F1"]); w.writerows(rows)
    if a.sweep:
        res = []
        for c in CLOSES:
            for ar in AREAS:
                for g in GROWS:
                    tp5 = tp3 = npred = ngt = 0
                    for gt_al, per_close in sweep_data:
                        st, shape = per_close[c]
                        pr = boxes_from(st, shape, ar, g)
                        tp5 += len(match(gt_al, pr, 0.5)); tp3 += len(match(gt_al, pr, 0.3)); npred += len(pr); ngt += len(gt_al)
                    p, rc = tp5 / max(npred, 1), tp5 / max(ngt, 1)
                    res.append([c, ar, g, npred, tp5, tp3, p, rc, 2 * p * rc / max(p + rc, 1e-9)])
        res.sort(key=lambda x: -x[-1])
        with open(a.out / "pipeline_val_sweep.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["close", "min_area", "grow", "pred", "tp_iou0.5", "tp_iou0.3", "P_0.5", "R_0.5", "F1_0.5"])
            w.writerows([[*r[:6], *[f"{x:.4f}" for x in r[6:]]] for r in res])
        print("\nTop 8 to hop tham so tren val (xep theo F1 o IoU 0,5):")
        print(f"{'close':>6}{'min_area':>9}{'grow':>6}{'pred':>7}{'TP0.5':>7}{'P':>8}{'R':>8}{'F1':>8}")
        for r in res[:8]:
            print(f"{r[0]:>6}{r[1]:>9}{r[2]:>6}{r[3]:>7}{r[4]:>7}{r[6]:>8.3f}{r[7]:>8.3f}{r[8]:>8.3f}")
        print("-> ghi to hop dau tien vao XOR_PARAMS trong tools/pipeline.py")
    if geo:
        g = np.median(np.asarray(geo), axis=0)
        print(f"\nCap chong lan (IoU>=0,1, bien the new, n={len(geo)}), trung vi cua box du doan so voi GT:")
        print(f"  rong x{g[0]:.2f}, cao x{g[1]:.2f}; lech tam ngang {g[2]:+.2f}, doc {g[3]:+.2f} (don vi = kich thuoc GT)")
    print(f"{secs:.2f} s/anh (gom doc anh, SIFT, XOR, phan loai)")
    for n, e in failed[:5]:
        print("Can chinh that bai:", n, e)

    if cls_model is not None and cls_n:
        print(f"\nPhan loai tren box tim dung (IoU>={a.iou_cls}, bien the {a.variant}, cat tu anh {a.crop_from}): {cls_ok}/{cls_n} = {cls_ok / cls_n:.3f}")
        print(f"Kiem tra: phan loai thang tren box NHAN (anh goc): {gt_ok}/{gt_n} = {gt_ok / max(gt_n, 1):.3f}")
        print("Ma tran nham lan (hang = that, cot = du doan), thu tu:", names)
        print(conf_mat)
        with open(a.out / f"pipeline_{a.split}_cls.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred", *names]); [w.writerow([names[i], *conf_mat[i]]) for i in range(len(names))]
            w.writerow(["accuracy", f"{cls_ok / cls_n:.4f}", f"n={cls_n}"])
    print(f"Ket qua: {a.out}")


if __name__ == "__main__":
    main()
