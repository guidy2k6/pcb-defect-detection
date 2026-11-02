import argparse
from datetime import datetime
from pathlib import Path

import ultralytics
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "configs" / "train_params.md"


def log_run(a):
    """Ghi tham so vao cung bang voi train.py (imgsz = kich thuoc anh cat)."""
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
    ap.add_argument("--name", default="pcb_cls_board")
    ap.add_argument("--model", default="yolov8n-cls.pt")
    ap.add_argument("--data", default=str(REPO / "dataset" / "cls"), help="thu muc co train/ val/ test/ (collect_crops.py tao)")
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--test", action="store_true", help="sau khi train: danh gia tren test (chi lam mot lan)")
    a = ap.parse_args()

    log_run(a)
    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, batch=a.batch, epochs=a.epochs, patience=a.patience,
        workers=a.workers, seed=a.seed, deterministic=True, device=a.device,
        project=str(REPO / "runs" / "classify"), name=a.name, exist_ok=False,
    )
    if a.test:
        best = Path(model.trainer.best)
        m = YOLO(str(best)).val(data=a.data, split="test", imgsz=a.imgsz, batch=a.batch, device=a.device, workers=0,
                                plots=True, project=str(REPO / "results"), name=f"{a.name}_test", exist_ok=True)
        print(f"\nDo chinh xac phan loai tren anh cat theo nhan (test): top1 = {m.top1:.4f}, top5 = {m.top5:.4f}")
        print("Day la can tren: bo phan loai chua nhan hop do buoc loc tim ra (xem inference.py --mode xor).")


if __name__ == "__main__":
    main()
