import argparse
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "configs" / "train_params.md"


# Khoa Ultralytics cho phep dat trong file --aug (go sai ten se bi Ultralytics bo qua hoac bao loi muon)
ULTRALYTICS_AUG_KEYS = {"flipud", "fliplr", "degrees", "perspective", "hsv_h", "hsv_s", "hsv_v",
                        "translate", "scale", "shear", "mosaic", "mixup", "close_mosaic"}


def load_aug(path):
    """Doc file augmentation. Tra ve (kwargs cho model.train, list spec albumentations)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    extra = set(cfg) - {"ultralytics", "albumentations"}
    if extra:
        raise SystemExit(f"{path}: khoa la {sorted(extra)}; chi co 'ultralytics' va 'albumentations'.")
    kw = dict(cfg.get("ultralytics") or {})
    bad = set(kw) - ULTRALYTICS_AUG_KEYS
    if bad:
        raise SystemExit(f"{path}: tham so ultralytics khong ho tro: {sorted(bad)}. Cho phep: {sorted(ULTRALYTICS_AUG_KEYS)}")
    spec = list(cfg.get("albumentations") or [])
    for s in spec:
        if "name" not in s:
            raise SystemExit(f"{path}: moi muc albumentations can 'name': {s}")
    return kw, spec


def build_albumentations(spec):
    """Dung danh sach transform Albumentations tu spec. Thieu thu vien thi dung han (khong train am tham thieu aug)."""
    try:
        import albumentations as A
    except ImportError:
        raise SystemExit("Can albumentations cho augmentation mo/xoay 90 do: pip install albumentations")
    out = []
    for s in spec:
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in s.items() if k != "name"}
        if not hasattr(A, s["name"]):
            raise SystemExit(f"Albumentations khong co transform '{s['name']}'")
        out.append(getattr(A, s["name"])(**kw))
    return out


def log_run(a):
    import ultralytics

    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text(
            "# Tham so cac lan huan luyen\n\n"
            "| Ngay gio | Run | Model | Data | imgsz | batch | epochs | patience | seed | ultralytics | aug |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"| {datetime.now():%Y-%m-%d %H:%M} | {a.name} | {a.model} | {a.data} | {a.imgsz} | "
                f"{a.batch} | {a.epochs} | {a.patience} | {a.seed} | {ultralytics.__version__} | "
                f"{a.aug or 'mac dinh'} |\n")


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
    ap.add_argument("--aug", default=None, help="file augmentation, vd configs/aug_robust.yaml; bo trong = mac dinh Ultralytics")
    a = ap.parse_args()

    aug_kw, aug_extra = {}, {}
    if a.aug:  # doc va kiem tra truoc khi ghi log, de loi cau hinh khong de lai dong log cua lan chay hong
        aug_kw, spec = load_aug(a.aug)
        if spec:
            aug_extra["augmentations"] = build_albumentations(spec)

    from ultralytics import YOLO

    log_run(a)
    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, batch=a.batch, epochs=a.epochs, patience=a.patience,
        workers=a.workers, seed=a.seed, deterministic=True, device=a.device,
        project=str(REPO / "runs" / "detect"), name=a.name, exist_ok=False,
        **aug_kw, **aug_extra,
    )


if __name__ == "__main__":
    main()