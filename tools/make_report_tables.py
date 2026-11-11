"""Tao reports/bang_so_lieu.md tu results/ de bao cao, README, slide dung chung mot nguon.

Chay tu thu muc goc repo:  python tools\\make_report_tables.py
"""
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
RUN = "pcb_v8s_board"          # doi neu dung mo hinh khac
f3 = lambda x: f"{x:.3f}".replace(".", ",")


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def overall(cm):
    labels = [i for i in cm.index if not str(i).startswith("background")]
    tp = sum(cm.loc[c, c] for c in labels)
    gt = cm.loc[labels].sum().sum()
    pred = cm[labels].sum().sum()
    missed = cm.loc[labels, "missed"].sum()
    p, r = tp / pred, tp / gt
    return p, r, 2 * p * r / (p + r), int(pred - tp), int(missed)


out = ["# Bang so lieu (tu sinh, khong sua tay)\n"]

s = pd.read_csv(RES / "summary.csv")
s = s[s.run == RUN][["eval_set", "P", "R", "mAP50", "mAP50-95"]].copy()
for c in ("P", "R", "mAP50", "mAP50-95"):
    s[c] = s[c].map(f3)
out += ["## 1. Val va test (conf 0,25)\n", md(s), ""]

rows = []
for sp in ("val", "test"):
    k = pd.read_csv(RES / f"{RUN}_{sp}_errors" / "false_pos_kinds.csv")
    rows.append([sp, int(k.false_pos.sum()), int(k.trung_gt_da_ghep.sum()),
                 int(k["lech_iou_0.1_den_0.5"].sum()), int(k["nen_iou_duoi_0.1"].sum())])
out += ["## 2. Box thua theo loai\n",
        md(pd.DataFrame(rows, columns=["tap", "FP", "trung", "lech", "nen"])), ""]

sweep = []
for d in RES.glob("val_conf_*_val_errors"):
    m = re.fullmatch(r"val_conf_([0-9.]+)_val_errors", d.name)
    if m and (d / "confusion_true_x_pred.csv").exists():
        p, r, f1, fp, miss = overall(pd.read_csv(d / "confusion_true_x_pred.csv", index_col=0))
        sweep.append([float(m.group(1)), f3(p), f3(r), f3(f1), fp, miss])
if sweep:
    sw = pd.DataFrame(sorted(sweep), columns=["conf", "P", "R", "F1", "box_sai", "bo_sot"])
    out += ["## 3. Quet conf tren val\n", md(sw), ""]

t = RES.parent / "configs" / "thresholds.json"
if t.exists():
    cfg = json.loads(t.read_text(encoding="utf-8"))
    out += ["## 4. Nguong da chon (tren val)\n", f"Nguong chung: {cfg['global']:g}\n",
            md(pd.DataFrame(list(cfg["per_class"].items()), columns=["lop", "conf"])), ""]

d = pd.read_csv(RES / f"{RUN}_test_errors" / "false_pos_list.csv")
d["bo"] = d.image.str.extract(r"^(\d+)_")[0]
d["cx"] = ((d.x1 + d.x2) / 2 // 30 * 30).astype(int)
d["cy"] = ((d.y1 + d.y2) / 2 // 30 * 30).astype(int)
g = (d.groupby(["bo", "pred_class", "cx", "cy"])
       .agg(anh=("image", "nunique"), conf_tb=("conf", "mean")).reset_index())
hot = g[g.anh >= 5].sort_values(["bo", "anh"], ascending=[True, False]).copy()
n_hot = int(d.merge(hot[["bo", "pred_class", "cx", "cy"]]).shape[0])
hot["conf_tb"] = hot.conf_tb.map(lambda x: f"{x:.2f}")
out += ["## 5. Box thua tren test theo vi tri (tam, lam tron 30 px; >= 5 anh)\n", md(hot), "",
        f"{n_hot}/{len(d)} FP nam o {len(hot)} vi tri lap lai; {len(g) - len(hot)} vi tri con lai la le te.\n"]

dst = REPO / "reports" / "bang_so_lieu.md"
dst.parent.mkdir(exist_ok=True)
dst.write_text("\n".join(out), encoding="utf-8")
print("Da ghi", dst)