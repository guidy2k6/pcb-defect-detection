import glob, os
import pandas as pd

rows = []
for d in sorted(glob.glob("results/*conf_*")):
    f = os.path.join(d, "error_per_class.csv")
    if os.path.exists(f):
        e = pd.read_csv(f)
        gt   = e["gt"].sum()
        miss = e["missed"].sum()
        fp   = e["false_pos_by_pred_class"].sum()
        rows.append((
            os.path.basename(d),
            gt,
            miss,
            fp,
            round((gt - miss) / gt, 3) if gt else 0.0,
        ))

print(pd.DataFrame(rows, columns=["run", "gt", "bo_sot", "FP", "recall"]))