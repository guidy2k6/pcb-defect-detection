"""Runtime configuration for the demo.

Every value can be set with an environment variable (PCB_*) and is overridable from the command line in demo/app.py.
Nothing here imports torch or ultralytics, so it is cheap to import and easy to test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Class ids in the YOLO label files follow alphabetical order. These are the names the detector weights must carry.
DETECTOR_CLASSES = {0: "Missing Hole", 1: "Mouse Bite", 2: "Open Circuit", 3: "Short", 4: "Spur", 5: "Spurious Copper"}
# Names that older detector weights carried before tools/fix_class_names.py patched them (a known permutation bug).
STALE_DETECTOR_CLASSES = {0: "Mouse Bite", 1: "Spur", 2: "Missing Hole", 3: "Short", 4: "Open Circuit", 5: "Spurious Copper"}
CLASS_DISPLAY = tuple(DETECTOR_CLASSES.values())

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    det_weights: Path = REPO / "models" / "best.pt"
    cls_weights: Path = REPO / "models" / "best_cls.pt"
    hf_repo: str | None = None  # Hugging Face model repo holding best.pt / best_cls.pt
    hf_det_file: str = "best.pt"
    hf_cls_file: str = "best_cls.pt"
    device: str = "auto"  # auto | cpu | 0 | cuda:0
    imgsz: int = 1024  # the size the detector was trained and evaluated at
    default_conf: float = 0.25  # chọn trên val: recall cao nhất, tổng box sai + bỏ sót thấp nhất (báo cáo mục 6.5)
    iou_nms: float = 0.70
    max_upload_mb: int = 30
    min_side_px: int = 256
    max_display_side: int = 1600  # annotated images are downscaled to this for the browser
    results_dir: Path = REPO / "results"
    examples_dir: Path = REPO / "demo" / "examples"
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    auth: tuple[str, str] | None = None  # (user, password)
    title: str = "Kiểm tra lỗi bảng mạch in (PCB)"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.det_weights = Path(_env("PCB_DET_WEIGHTS", str(s.det_weights)))
        s.cls_weights = Path(_env("PCB_CLS_WEIGHTS", str(s.cls_weights)))
        s.hf_repo = _env("HF_MODEL_REPO")
        s.hf_det_file = _env("PCB_HF_DET_FILE", s.hf_det_file)
        s.hf_cls_file = _env("PCB_HF_CLS_FILE", s.hf_cls_file)
        s.device = _env("PCB_DEVICE", s.device)
        s.imgsz = _env_int("PCB_IMGSZ", s.imgsz)
        s.default_conf = _env_float("PCB_CONF", s.default_conf)
        s.max_upload_mb = _env_int("PCB_MAX_UPLOAD_MB", s.max_upload_mb)
        s.host = _env("PCB_HOST", s.host)
        s.port = _env_int("PCB_PORT", s.port)
        s.share = _env("PCB_SHARE", "0") in ("1", "true", "True")
        credentials = _env("PCB_AUTH")  # "user:password"
        if credentials and ":" in credentials:
            user, _, password = credentials.partition(":")
            s.auth = (user, password)
        return s
