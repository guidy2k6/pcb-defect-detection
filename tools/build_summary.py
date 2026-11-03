"""Gom ket qua cac lan chay thanh results/summary.csv (run, cach chia, mo hinh, imgsz, P, R, mAP50, mAP50-95, ms/anh).

Doc results/<run>_val/metrics_per_class.csv va results/<run>_test/metrics_per_class.csv (do eval.py ghi), dong 'all'.
ms/anh lay tu results/<run>_<tap>/speed.txt (eval.py ghi); lan chay cu chua co thi truyen tay:
    python tools/build_summary.py --ms pcb_v8n_random_val=13.6 pcb_v8s_val=24.9
"""
import argparse
import csv
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def run_info(run, results):
    """Suy ra cach chia, mo hinh, imgsz tu ten run (+ args.yaml neu co). pcb_v8s (chay 06/10) la chia ngau nhien."""
    base = run.rsplit("_", 1)[0] if run.endswith(("_val", "_test")) else run
    cfg = {}
    args = results / base / "args.yaml"
    if args.exists():
        cfg = yaml.safe_load(args.read_text(encoding="utf-8"))
    split = "Theo bo" if "board" in base or "data_board" in str(cfg.get("data", "")) else "Ngau nhien"
    m = re.search(r"v8([nsmlx])", base) or re.search(r"yolov8([nsmlx])", str(cfg.get("model", "")))
    model = f"yolov8{m.group(1)}" if m else {"pcb_best_640": "yolov8s"}.get(base, "?")
    return base, split, model, cfg.get("imgsz", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--ms", nargs="*", default=[], help="<thu_muc_ket_qua>=<ms>, vd pcb_v8s_val=24.9")
    ap.add_argument("--imgsz", type=int, default=1024, help="dung khi khong co args.yaml")
    a = ap.parse_args()
    manual = dict(kv.split("=", 1) for kv in a.ms)

    rows = []
    for csv_path in sorted(a.results.glob("*_*/metrics_per_class.csv")):
        d = csv_path.parent
        if not d.name.endswith(("_val", "_test")):
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            allrow = next((r for r in csv.DictReader(f) if r["class"] == "all"), None)
        if allrow is None:
            continue
        run, split, model, imgsz = run_info(d.name, a.results)
        speed = d / "speed.txt"
        ms = manual.get(d.name) or (speed.read_text().strip() if speed.exists() else "")
        rows.append([run, split, d.name.rsplit("_", 1)[1], model, imgsz or a.imgsz,
                     allrow["precision"], allrow["recall"], allrow["AP50"], allrow["AP50-95"], ms])

    out = a.results / "summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "split", "eval_set", "model", "imgsz", "P", "R", "mAP50", "mAP50-95", "ms_per_image"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")
    missing = [r[0] + "_" + r[2] for r in rows if not r[-1]]
    if missing:
        print(f"\nThieu ms/anh: {missing} -> them --ms <thu_muc>=<ms>")
    print(f"Ket qua: {out}")


if __name__ == "__main__":
    main()
