import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from yolo_utils import label_path, load_split, read_labels

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--name", required=True, help="vd pcb_v8n_random")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--force", action="store_true", help="cho phep chay lai test (khong nen)")
    a = ap.parse_args()

    test_csv = REPO / "results" / "test_metrics.csv"
    if a.split == "test" and test_csv.exists() and not a.force:
        raise SystemExit(f"Test da duoc chay ({test_csv} ton tai). Test chi chay MOT lan. "
                         "Dung --force neu that su can va ghi ly do vao configs/train_params.md.")

    names, images = load_split(a.data, a.split)
    inst = {i: 0 for i in names}
    img_with = {i: set() for i in names}
    for im in images:
        for c, *_ in read_labels(im):
            inst[c] += 1
            img_with[c].add(im)

    out_name = f"{a.name}_{a.split}"
    m = YOLO(a.weights).val(
        data=a.data, split=a.split, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=0,
        plots=True, project=str(REPO / "results"), name=out_name, exist_ok=True,
    )

    rows = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append([names[int(c)], len(img_with[int(c)]), inst[int(c)], p, r, ap50, ap])
    rows.append(["all", len(images), sum(inst.values()), m.box.mp, m.box.mr, m.box.map50, m.box.map])

    out_dir = REPO / "results" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["class", "images", "instances", "precision", "recall", "AP50", "AP50-95"]
    with open(out_dir / "metrics_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[r[0], r[1], r[2], *[f"{x:.4f}" for x in r[3:]]] for r in rows])

    speed = m.speed  # ms/anh: preprocess, inference, loss, postprocess
    ms = speed["preprocess"] + speed["inference"] + speed["postprocess"]
    if a.split == "test":
        with open(test_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["weights", "data", "imgsz", "precision", "recall", "AP50", "AP50-95", "ms_per_image"])
            w.writerow([a.weights, a.data, a.imgsz, *[f"{x:.4f}" for x in rows[-1][3:]], f"{ms:.2f}"])
        print(f"Da ghi {test_csv}")

    print(f"\n{'class':<18}{'imgs':>6}{'inst':>7}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>9}")
    for r in rows:
        print(f"{r[0]:<18}{r[1]:>6}{r[2]:>7}{r[3]:>8.3f}{r[4]:>8.3f}{r[5]:>8.3f}{r[6]:>9.3f}")
    print(f"\nThoi gian: {ms:.2f} ms/anh (preprocess+inference+postprocess)")
    print(f"Ket qua: {out_dir}")


if __name__ == "__main__":
    main()