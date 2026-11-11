"""Chon nguong conf rieng tung lop tren VAL va ghi configs/thresholds.json.

Doc results/val_conf_<conf>_val_errors/confusion_true_x_pred.csv (tao boi
analyze_errors.py --split val --name val_conf_<conf>). Chay tu thu muc goc repo:
  python tools\\select_thresholds.py
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def class_metrics(cm):
    """cm: hang = lop that (+ background), cot = lop du doan (+ missed)."""
    out = {}
    for c in [i for i in cm.index if not str(i).startswith("background")]:
        tp = cm.loc[c, c]
        gt = cm.loc[c].sum()   # gom ca cot missed
        pred = cm[c].sum()     # gom ca hang background (FP nen)
        p = tp / pred if pred else 0.0
        r = tp / gt if gt else 0.0
        out[c] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "configs" / "thresholds.json"))
    ap.add_argument("--min-recall", type=float, default=0.90,
                    help="chi xet nguong co recall lop >= gia tri nay (neu co)")
    a = ap.parse_args()

    runs = {}
    for d in Path(a.results).glob("val_conf_*_val_errors"):
        m = re.fullmatch(r"val_conf_([0-9.]+)_val_errors", d.name)
        f = d / "confusion_true_x_pred.csv"
        if m and f.exists():
            runs[float(m.group(1))] = class_metrics(pd.read_csv(f, index_col=0))
    if not runs:
        raise SystemExit("Khong thay results/val_conf_*_val_errors. Chay quet conf truoc.")
    confs = sorted(runs)
    classes = list(runs[confs[0]])

    per_class, table = {}, {}
    for c in classes:
        rows = [(cf, *runs[cf][c]) for cf in confs]          # (conf, P, R, F1)
        ok = [r for r in rows if r[2] >= a.min_recall] or [max(rows, key=lambda r: r[2])]
        best = max(ok, key=lambda r: (round(r[3], 4), -r[0]))  # F1 cao nhat, hoa thi conf thap
        per_class[c] = best[0]
        table[c] = {f"{r[0]:g}": {"P": round(r[1], 3), "R": round(r[2], 3), "F1": round(r[3], 3)} for r in rows}

    macro = {cf: sum(runs[cf][c][2] for c in classes) / len(classes) for cf in confs}
    glob_conf = max(confs, key=lambda cf: (round(macro[cf], 4), -cf))

    cfg = {"chon_tren": "val", "confs_da_thu": confs, "min_recall": a.min_recall,
           "global": glob_conf, "per_class": per_class, "val_metrics": table}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'lop':18s}" + "".join(f"conf {cf:<5g}" for cf in confs) + " -> chon")
    for c in classes:
        cells = "".join(f"P{runs[cf][c][0]:.2f}/R{runs[cf][c][1]:.2f} " for cf in confs)
        print(f"{c:18s}{cells} -> {per_class[c]:g}")
    print(f"\nNguong chung (macro-F1 cao nhat): {glob_conf:g}\nDa ghi {a.out}")


if __name__ == "__main__":
    main()