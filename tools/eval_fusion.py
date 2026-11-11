"""Danh gia ket hop YOLO + XOR: loc box YOLO bang ban do khac biet XOR (xem tools/fusion.py).

    # 1) Chon min_px tren VAL (khong cham vao test); ghi configs/fusion.json va results/fusion_val_sweep.csv
    python tools/eval_fusion.py --dataset PCB_DATASET --weights models/best.pt --split val --sweep --device 0
    # 2) Chay TEST dung mot lan voi min_px da chon (doc tu configs/fusion.json)
    python tools/eval_fusion.py --dataset PCB_DATASET --weights models/best.pt --split test --device 0

So sanh hai cach tren CUNG nguong conf theo lop (configs/thresholds.json): YOLO thuan va YOLO + loc XOR.
Cach tinh: khop cung lop, IoU >= 0,5, P/R gop tren tat ca box (khong phai trung binh theo lop cua Ultralytics).
Anh khong can chinh duoc voi bo mau: giu nguyen box YOLO (khong xac minh duoc) va duoc dem rieng.
Ket qua: results/fusion_<split>.csv (tong, theo lop). Test khong ghi de neu da co, tru khi --force.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from baseline_xor import IMAGE_EXTENSIONS
from crop_utils import class_folder
from eval_pipeline import read_gt
from fusion import box_to_aligned, change_mask, evidence, keep_flags, match_cls, prf
from pipeline import XOR_PARAMS, xor_detect

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
CANDIDATES = (0, 10, 25, 50, 100, 200, 400, 800)  # min_px thu tren val; 0 = khong loc (YOLO thuan)
MAX_RECALL_DROP = 0.01  # chi nhan min_px neu recall val giam khong qua 1 diem so voi YOLO thuan


def totals(records, min_px, min_valid):
    """Cong tp/fp/fn (tong va theo lop) cua toan bo anh voi quy tac loc min_px."""
    tot = np.zeros(3, int)
    per = {c: np.zeros(3, int) for c in CLASSES}
    for r in records:
        keep = keep_flags(r["n_px"], r["v_frac"], min_px, min_valid) if len(r["pr"]) else np.zeros(0, bool)
        pr, pc = r["pr"][keep], r["pr_cls"][keep]
        tot += match_cls(r["gt"], r["gt_cls"], pr, pc)
        for c in CLASSES:
            gm, pm = r["gt_cls"] == c, pc == c
            per[c] += match_cls(r["gt"][gm], r["gt_cls"][gm], pr[pm], pc[pm])
    return tot, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="thu muc PCB_DATASET (co images, Annotations, PCB_USED)")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=REPO / "configs" / "board_split.json")
    ap.add_argument("--thresholds", type=Path, default=REPO / "configs" / "thresholds.json")
    ap.add_argument("--fusion-config", type=Path, default=REPO / "configs" / "fusion.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--sweep", action="store_true", help="chon min_px tren val (cam tren test)")
    ap.add_argument("--pad", type=int, default=8, help="le (px) quanh box khi dem pixel khac biet")
    ap.add_argument("--min-valid", type=float, default=0.5, help="duoi ti le vung hop le nay thi khong xac minh, giu box")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--force", action="store_true", help="cho phep ghi de results/fusion_test.csv")
    a = ap.parse_args()

    if a.sweep and a.split != "val":
        raise SystemExit("--sweep chi chay tren val: khong chon tham so bang tap test.")
    out_csv = a.out / f"fusion_{a.split}.csv"
    if a.split == "test" and out_csv.exists() and not a.force:
        raise SystemExit(f"{out_csv} da ton tai. Test chi danh gia mot lan; them --force neu that su muon chay lai.")
    if a.split == "test":
        if not a.fusion_config.exists():
            raise SystemExit(f"Chua co {a.fusion_config}: chay --sweep tren val truoc.")
        cfg = json.loads(a.fusion_config.read_text(encoding="utf-8"))
        a.pad, a.min_valid = cfg["pad"], cfg["min_valid"]  # tham so da chot tren val, khong cho doi o test

    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    names = {int(k): class_folder(v) for k, v in model.names.items()}
    thr_cfg = json.loads(a.thresholds.read_text(encoding="utf-8"))["per_class"]
    thr = np.array([thr_cfg[model.names[i]] for i in sorted(model.names)])

    boards = {str(b) for b in json.loads(a.manifest.read_text(encoding="utf-8"))[a.split]}
    templates = {p.stem: p for p in (a.dataset / "PCB_USED").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    xmls = {p.stem.casefold(): p for p in (a.dataset / "Annotations").rglob("*.xml")}
    images = sorted(p for p in (a.dataset / "images").rglob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.split("_", 1)[0] in boards)
    if a.limit:
        images = images[:a.limit]
    if not images:
        raise SystemExit(f"Khong co anh cua cac bo {sorted(boards)} trong {a.dataset / 'images'}")

    records, failed = [], []
    for k, img in enumerate(images):
        gt, gt_names, _ = read_gt(xmls[img.stem.casefold()])
        res = model.predict(source=str(img), imgsz=a.imgsz, conf=float(thr.min()), device=a.device, verbose=False, batch=1)[0]
        xyxy = res.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        ok = conf >= thr[cls]
        xyxy, cls = xyxy[ok], cls[ok]
        n_px, v_frac = np.zeros(len(xyxy), int), np.zeros(len(xyxy))  # mac dinh: khong xac minh -> giu
        try:
            r = xor_detect(templates[img.stem.split("_", 1)[0]], img)
            mask = change_mask(r["xor_raw"], r["valid"], XOR_PARAMS["close"])
            n_px, v_frac = evidence(box_to_aligned(xyxy, r["H"]), mask, r["valid"], a.pad)
        except Exception as e:  # can chinh that bai
            failed.append((img.name, str(e)))
        records.append({"gt": gt, "gt_cls": np.array(gt_names), "pr": xyxy, "pr_cls": np.array([names[c] for c in cls]),
                        "n_px": n_px, "v_frac": v_frac})
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(images)}", flush=True)

    base_tot, base_per = totals(records, 0, a.min_valid)
    bp, br, bf = prf(*base_tot)
    print(f"\nTap {a.split}: {len(images)} anh, bo {sorted(boards)}, can chinh that bai: {len(failed)}")
    print(f"{'min_px':>7}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}")
    rows = []
    for m in (CANDIDATES if a.sweep else (json.loads(a.fusion_config.read_text(encoding='utf-8'))["min_px"],)):
        tot, _ = totals(records, m, a.min_valid)
        p, rc, f1 = prf(*tot)
        rows.append([m, *tot.tolist(), p, rc, f1])
        print(f"{m:>7}{tot[0]:>6}{tot[1]:>6}{tot[2]:>6}{p:>8.3f}{rc:>8.3f}{f1:>8.3f}")
    a.out.mkdir(parents=True, exist_ok=True)

    if a.sweep:
        with open(a.out / "fusion_val_sweep.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["min_px", "tp", "fp", "fn", "P", "R", "F1"])
            w.writerows([[*r[:4], *[f"{x:.4f}" for x in r[4:]]] for r in rows])
        ok = [r for r in rows if r[0] > 0 and r[5] >= br - MAX_RECALL_DROP]
        best = max(ok, key=lambda r: (r[6], -r[0])) if ok else None
        if best is None or best[6] <= bf:
            print("\nKhong co min_px nao tot hon YOLO thuan tren val (voi rang buoc recall); khong ghi configs/fusion.json.")
            return
        a.fusion_config.write_text(json.dumps(
            {"chon_tren": "val", "min_px": best[0], "pad": a.pad, "min_valid": a.min_valid, "max_recall_drop": MAX_RECALL_DROP,
             "val_yolo_only": {"P": round(bp, 4), "R": round(br, 4), "F1": round(bf, 4)},
             "val_fused": {"P": round(best[4], 4), "R": round(best[5], 4), "F1": round(best[6], 4)}},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nChon min_px={best[0]} (F1 {bf:.3f} -> {best[6]:.3f}); da ghi {a.fusion_config}")
        return

    min_px = rows[0][0]
    tot, per = totals(records, min_px, a.min_valid)
    fp_drop, tp_lost = int(base_tot[1] - tot[1]), int(base_tot[0] - tot[0])
    print(f"\nYOLO thuan : TP {base_tot[0]}  FP {base_tot[1]}  FN {base_tot[2]}  P {bp:.3f}  R {br:.3f}  F1 {bf:.3f}")
    p, rc, f1 = prf(*tot)
    print(f"YOLO + XOR : TP {tot[0]}  FP {tot[1]}  FN {tot[2]}  P {p:.3f}  R {rc:.3f}  F1 {f1:.3f}   (min_px={min_px})")
    print(f"Loai duoc {fp_drop} box thua, mat {tp_lost} box dung. Anh can chinh that bai (giu nguyen box): {len(failed)}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "class", "tp", "fp", "fn", "P", "R", "F1"])
        for label, t, pc in (("yolo", base_tot, base_per), ("yolo+xor", tot, per)):
            w.writerow([label, "ALL", *t.tolist(), *[f"{x:.4f}" for x in prf(*t)]])
            for c in CLASSES:
                w.writerow([label, c, *pc[c].tolist(), *[f"{x:.4f}" for x in prf(*pc[c])]])
    print(f"Ket qua: {out_csv}")


if __name__ == "__main__":
    main()