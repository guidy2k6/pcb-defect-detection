import argparse
from datetime import datetime
from pathlib import Path

import ultralytics
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "configs" / "train_params.md"


def log_run(a):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text(
            "# Tham so cac lan huan luyen\n\n"
            "| Ngay gio | Run | Model | Data | imgsz | batch | epochs | patience | seed | ultralytics |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"| {datetime.now():%Y-%m-%d %H:%M} | {a.name} | {a.model} | {a.data} | {a.imgsz} | "
                f"{a.batch} | {a.epochs} | {a.patience} | {a.seed} | {ultralytics.__version__} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="ten run, vd pcb_v8n_random")
    ap.add_argument("--model", required=True, help="yolov8n.pt hoac yolov8s.pt")
    ap.add_argument("--data", required=True, help="file yaml du lieu")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    log_run(a)
    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, batch=a.batch, epochs=a.epochs, patience=a.patience,
        workers=a.workers, seed=a.seed, deterministic=True, device=a.device,
        project=str(REPO / "runs" / "detect"), name=a.name, exist_ok=False,
    )


if __name__ == "__main__":
    main()