from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_split(data_yaml, split):
    """Doc data.yaml -> (names {id: ten}, danh sach anh cua tap `split`)."""
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}

    base = Path(cfg["path"]) if cfg.get("path") else data_yaml.parent
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    entries = cfg[split]
    entries = [entries] if isinstance(entries, str) else list(entries)

    images = []
    for e in entries:
        p = Path(e)
        if not p.is_absolute():
            p = base / p
        if p.is_dir():
            images += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.suffix == ".txt":
            images += [Path(l.strip()) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        elif p.is_file():
            images.append(p)
        else:
            raise FileNotFoundError(f"Khong thay {p} (tu muc '{split}' trong {data_yaml})")
    if not images:
        raise SystemExit(f"Tap '{split}' khong co anh nao.")
    return names, images


def label_path(img: Path) -> Path:
    """.../images/x/a.jpg -> .../labels/x/a.txt (quy uoc cua Ultralytics)."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_labels(img: Path):
    """Tra ve list (class_id, xc, yc, w, h) chuan hoa; rong neu khong co file nhan."""
    p = label_path(img)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) >= 5:
            rows.append((int(float(v[0])), *map(float, v[1:5])))
    return rows