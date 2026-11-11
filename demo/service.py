"""Inference service behind the Gradio UI. No gradio import here, so everything is unit-testable.

Four analyses are exposed by PCBInspector:
  detect()          one image  -> YOLOv8s detector (single stage)
  xor_inspect()     image + defect-free template of the same board -> SIFT alignment, XOR, crop classifier (the paper's route)
  cross_check()     both of the above on the same pair, with a simple agreement rule between them (illustrative only)
  fusion_check()    YOLO boxes verified against the XOR change map of the same pair (the evaluated fusion, tools/fusion.py)

Design notes
  * Models load lazily and are validated: detector weights that still carry the old permuted class names are rejected
    with the exact command that fixes them (tools/fix_class_names.py).
  * A single lock serialises model calls; CUDA out-of-memory falls back to CPU once and the fallback is reported.
  * Uploaded files are validated (type, size, decodable, minimum side) and staged under ASCII names before OpenCV reads
    them, because cv2.imread cannot open non-ASCII paths on Windows.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import (CLASS_DISPLAY, DETECTOR_CLASSES, IMAGE_EXTENSIONS, REPO, STALE_DETECTOR_CLASSES, Settings)

log = logging.getLogger("pcb.demo")

# Okabe-Ito colour-blind-safe palette (RGB), one colour per defect class.
CLASS_RGB = {
    "Missing Hole": (0, 114, 178), "Mouse Bite": (230, 159, 0), "Open Circuit": (0, 158, 115),
    "Short": (213, 94, 0), "Spur": (204, 121, 167), "Spurious Copper": (140, 120, 20),
}
SOURCE_RGB = {"both": (0, 158, 115), "yolo": (0, 114, 178), "xor": (213, 94, 0)}
SOURCE_TEXT = {"both": "Cả hai hướng", "yolo": "Chỉ YOLO", "xor": "Chỉ XOR"}


class InputError(ValueError):
    """The user supplied something unusable (bad image, wrong size, missing template). Message is user-facing."""


class ModelError(RuntimeError):
    """Weights are missing or inconsistent. Message is user-facing."""


class AlignmentError(RuntimeError):
    """SIFT alignment between template and test image failed or looks wrong. Message is user-facing."""


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the test image
    source: str = "yolo"  # yolo | xor | both
    note: str = ""

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class Analysis:
    mode: str
    annotated: np.ndarray  # RGB, already downscaled for display
    detections: list[Detection]
    image_name: str
    image_shape: tuple[int, int]  # height, width of the test image
    params: dict
    timings_ms: dict[str, float]
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    extra_images: dict[str, np.ndarray] = field(default_factory=dict)
    crops: list[tuple[np.ndarray, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    score_kind: str = "detector"  # what Detection.score means in this mode

    def counts(self) -> dict[str, int]:
        out = {name: 0 for name in CLASS_DISPLAY}
        for d in self.detections:
            out[d.label] = out.get(d.label, 0) + 1
        return out


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------
def display_name(raw: str) -> str:
    """'open_circuit' -> 'Open Circuit' (the classifier uses folder-style names, the detector uses display names)."""
    return " ".join(part.capitalize() for part in raw.replace("_", " ").split())


def _np(x) -> np.ndarray:
    """Tensor or array-like -> numpy (keeps the service independent of torch for tests)."""
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def load_image(path, max_mb: int = 30, min_side: int = 256) -> np.ndarray:
    """Validate and decode an uploaded image to a BGR array."""
    if path is None or str(path) == "":
        raise InputError("Chưa chọn ảnh.")
    p = Path(str(path))
    if not p.is_file():
        raise InputError("Không tìm thấy tệp ảnh đã tải lên.")
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise InputError(f"Định dạng {p.suffix or '(không có)'} chưa được hỗ trợ. Dùng: {', '.join(IMAGE_EXTENSIONS)}.")
    size_mb = p.stat().st_size / 1e6
    if size_mb > max_mb:
        raise InputError(f"Ảnh nặng {size_mb:.1f} MB, vượt giới hạn {max_mb} MB.")
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("Không đọc được ảnh (tệp hỏng hoặc không phải ảnh).")
    if min(img.shape[:2]) < min_side:
        raise InputError(f"Ảnh quá nhỏ ({img.shape[1]}x{img.shape[0]} px). Cần cạnh ngắn tối thiểu {min_side} px.")
    return img


@contextmanager
def staged_files(*paths):
    """Copy files under ASCII names into a temp dir (cv2.imread cannot read non-ASCII paths on Windows)."""
    tmp = Path(tempfile.mkdtemp(prefix="pcb_stage_"))
    try:
        staged = []
        for i, src in enumerate(paths):
            dst = tmp / f"img{i}{Path(str(src)).suffix.lower() or '.png'}"
            shutil.copyfile(str(src), dst)
            staged.append(dst)
        yield staged
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(a, b, thr: float):
    """Greedy one-to-one matching by descending IoU. Returns (pairs, unmatched_a, unmatched_b)."""
    cand = sorted(((_iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for v, i, j in cand:
        if v < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return pairs, [i for i in range(len(a)) if i not in used_a], [j for j in range(len(b)) if j not in used_b]


# ----------------------------------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------------------------------
def _text_color(rgb) -> tuple[int, int, int]:
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def render(bgr: np.ndarray, items, max_side: int) -> np.ndarray:
    """Draw boxes on a downscaled copy. items = [(box_xyxy, text, rgb)]. Returns RGB for the browser."""
    h, w = bgr.shape[:2]
    ratio = min(1.0, max_side / max(h, w))
    out = cv2.resize(bgr, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA) if ratio < 1 else bgr.copy()
    k = max(1.0, out.shape[1] / 700)  # the browser shows the image about 700 px wide, so scale strokes and text to stay legible
    font, thick_text, thick_box = 0.55 * k, max(1, round(k)), max(2, round(2 * k))
    for box, text, rgb in items:
        x1, y1, x2, y2 = (int(round(v * ratio)) for v in box)
        color = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick_box, cv2.LINE_AA)
        if text:
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick_text)
            pad = max(3, round(3 * k))
            top = y1 - th - base - 2 * pad
            if top < 0:
                top = y2
            cv2.rectangle(out, (x1, top), (x1 + tw + 2 * pad, top + th + base + 2 * pad), color, -1)
            tc = _text_color(rgb)
            cv2.putText(out, text, (x1 + pad, top + th + pad), cv2.FONT_HERSHEY_SIMPLEX, font, (tc[2], tc[1], tc[0]), thick_text, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def evidence_crops(bgr: np.ndarray, dets: list[Detection], limit: int = 12, size: int = 256):
    """Close-up of each detection with surrounding context, for the gallery."""
    h, w = bgr.shape[:2]
    out = []
    for d in dets[:limit]:
        x1, y1, x2, y2 = d.box
        side = int(min(max(max(x2 - x1, y2 - y1) * 4, 96), h, w))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left = int(min(max(cx - side / 2, 0), w - side))
        top = int(min(max(cy - side / 2, 0), h - side))
        patch = bgr[top:top + side, left:left + side].copy()
        cv2.rectangle(patch, (int(x1 - left), int(y1 - top)), (int(x2 - left), int(y2 - top)), (0, 255, 255), 1, cv2.LINE_AA)
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)
        out.append((cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), f"{d.label} ({d.score:.2f})"))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# exports
# ----------------------------------------------------------------------------------------------------------------------
def detections_table(a: Analysis) -> list[list]:
    return [[i + 1, d.label, round(d.score, 3), int(d.box[0]), int(d.box[1]), int(d.box[2]), int(d.box[3]),
             int(d.width), int(d.height), SOURCE_TEXT.get(d.source, d.source) if a.mode == "cross" else "", d.note]
            for i, d in enumerate(a.detections)]


TABLE_HEADERS = ["#", "Loại lỗi", "Độ tin cậy", "x1", "y1", "x2", "y2", "Rộng (px)", "Cao (px)", "Nguồn", "Ghi chú"]


def to_csv(a: Analysis) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "label", "score", "x1", "y1", "x2", "y2", "width_px", "height_px", "source", "note"])
    for i, d in enumerate(a.detections):
        w.writerow([i + 1, d.label, f"{d.score:.4f}", *(f"{v:.1f}" for v in d.box), f"{d.width:.1f}", f"{d.height:.1f}", d.source, d.note])
    return buf.getvalue()


def to_json(a: Analysis, model_info: dict) -> str:
    return json.dumps({
        "request_id": a.request_id, "mode": a.mode, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image": {"name": a.image_name, "height": a.image_shape[0], "width": a.image_shape[1]},
        "params": a.params, "score_kind": a.score_kind, "timings_ms": a.timings_ms, "counts": a.counts(), "notes": a.notes,
        "model": model_info,
        "detections": [{"label": d.label, "score": round(d.score, 4), "box_xyxy": [round(v, 1) for v in d.box], "source": d.source,
                        "note": d.note} for d in a.detections],
    }, ensure_ascii=False, indent=2)


def write_exports(a: Analysis, model_info: dict, keep_seconds: int = 3600) -> tuple[str, str]:
    """Write CSV and JSON under the temp dir; also drop exports older than keep_seconds."""
    out_dir = Path(tempfile.gettempdir()) / "pcb_demo_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old in out_dir.glob("pcb_report_*"):
        try:
            if now - old.stat().st_mtime > keep_seconds:
                old.unlink()
        except OSError:
            pass
    csv_path = out_dir / f"pcb_report_{a.request_id}.csv"
    json_path = out_dir / f"pcb_report_{a.request_id}.json"
    # newline="" stops Windows from turning "\n" into "\r\n" twice; the BOM makes Excel open UTF-8 correctly
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(to_csv(a))
    with open(json_path, "w", encoding="utf-8", newline="") as f:
        f.write(to_json(a, model_info))
    return str(csv_path), str(json_path)


def load_reference_metrics(results_dir: Path) -> dict:
    """Read the measured results shipped in results/ so the UI can show them next to the live demo. Missing files are skipped."""
    def rows(path):
        try:
            with open(path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except OSError:
            return []
    return {
        "test": rows(results_dir / "test_metrics.csv"),
        "per_class": rows(results_dir / "pcb_v8s_board_test" / "metrics_per_class.csv"),
        "xor": [r for r in rows(results_dir / "pipeline_test.csv") if r.get("variant") == "new"],
        "summary": rows(results_dir / "summary.csv"),
        "fusion": [r for r in rows(results_dir / "fusion_test.csv") if r.get("class") == "ALL"],
    }


# ----------------------------------------------------------------------------------------------------------------------
# the service
# ----------------------------------------------------------------------------------------------------------------------
def _pipeline():
    """Import tools/pipeline.py lazily (it pulls SIFT code and is only needed for the template mode)."""
    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import pipeline  # noqa: WPS433
    import crop_utils  # noqa: WPS433
    return pipeline, crop_utils


def _fusion():
    """Import tools/fusion.py lazily (numpy + OpenCV only)."""
    _pipeline()  # puts tools/ on sys.path
    import fusion  # noqa: WPS433
    return fusion


def _fusion_params() -> dict:
    """min_px / pad / min_valid chosen on validation by tools/eval_fusion.py --sweep (configs/fusion.json); defaults otherwise."""
    params = {"min_px": 10, "pad": 8, "min_valid": 0.5}
    try:
        cfg = json.loads((REPO / "configs" / "fusion.json").read_text(encoding="utf-8"))
        params.update({k: cfg[k] for k in params if k in cfg})
    except (OSError, ValueError):
        pass
    return params


def _load_yolo(path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelError("Chưa cài ultralytics. Chạy: pip install -r requirements.txt") from exc
    try:
        return YOLO(str(path))
    except Exception as exc:  # noqa: BLE001 - corrupt or incompatible checkpoint
        raise ModelError(f"Không mở được trọng số {path}: {exc}") from exc


def check_detector_names(names: dict, path) -> None:
    """Reject detector weights whose class names are stale (permuted) or otherwise inconsistent with the label ids."""
    names = {int(k): v for k, v in dict(names).items()}
    if names == STALE_DETECTOR_CLASSES:
        raise ModelError(
            "Trọng số detector còn tên lớp cũ (bị hoán vị), kết quả sẽ gán sai tên lỗi. Sửa bằng: "
            f"python tools/fix_class_names.py det {path} models/best.pt")
    if names != DETECTOR_CLASSES:
        raise ModelError(f"Tên lớp của detector không khớp {list(DETECTOR_CLASSES.values())}: {names}")


def check_classifier_names(names: dict) -> None:
    got = {display_name(v) for v in dict(names).values()}
    if got != set(CLASS_DISPLAY):
        raise ModelError(f"Tên lớp của bộ phân loại không khớp 6 loại lỗi: {sorted(got)}")


class PCBInspector:
    def __init__(self, settings: Settings | None = None, detector=None, classifier=None):
        self.s = settings or Settings.from_env()
        self._det = detector
        self._cls = classifier
        self._lock = threading.RLock()
        self.device = self._resolve_device(self.s.device)
        self.fell_back_to_cpu = False

    # ---- models ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # torch missing or broken: CPU is the safe answer
            return "cpu"

    def _weights(self, kind: str) -> Path:
        local = self.s.det_weights if kind == "det" else self.s.cls_weights
        if local.is_file():
            return local
        if self.s.hf_repo:
            from huggingface_hub import hf_hub_download
            fname = self.s.hf_det_file if kind == "det" else self.s.hf_cls_file
            try:
                return Path(hf_hub_download(repo_id=self.s.hf_repo, filename=fname))
            except Exception as exc:
                raise ModelError(f"Không tải được {fname} từ Hugging Face repo {self.s.hf_repo}: {exc}") from exc
        which = "best.pt" if kind == "det" else "best_cls.pt"
        raise ModelError(f"Không tìm thấy trọng số {local}. Đặt tệp {which} vào models/ hoặc đặt biến HF_MODEL_REPO.")

    def has_classifier(self) -> bool:
        return self._cls is not None or self.s.cls_weights.is_file() or bool(self.s.hf_repo)

    @property
    def detector(self):
        with self._lock:
            if self._det is None:
                path = self._weights("det")
                model = _load_yolo(path)
                check_detector_names(model.names, path)
                self._det = model
                log.info("detector loaded from %s", path)
            return self._det

    @property
    def classifier(self):
        with self._lock:
            if self._cls is None:
                path = self._weights("cls")
                model = _load_yolo(path)
                check_classifier_names(model.names)
                self._cls = model
                log.info("classifier loaded from %s", path)
            return self._cls

    def _predict(self, model, source, **kwargs):
        """One serialised predict call; on CUDA OOM, free the cache and retry once on CPU (and stay on CPU)."""
        with self._lock:
            try:
                return model.predict(source, device=self.device, verbose=False, **kwargs)
            except Exception as exc:  # noqa: BLE001 - torch raises several types for OOM
                if self.device == "cpu" or not _is_oom(exc):
                    raise
                log.warning("CUDA out of memory, falling back to CPU: %s", exc)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                self.device = "cpu"
                self.fell_back_to_cpu = True
                return model.predict(source, device="cpu", verbose=False, **kwargs)

    def info(self) -> dict:
        """Facts for the model card and for the JSON export. Never loads weights just to describe them."""
        def sha(path: Path):
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                return h.hexdigest()[:12]
            except OSError:
                return None
        versions = {}
        for pkg in ("ultralytics", "torch", "cv2"):
            try:
                versions[pkg] = __import__(pkg).__version__
            except Exception:  # noqa: BLE001
                versions[pkg] = None
        return {
            "detector": {"file": str(self.s.det_weights), "available": self._det is not None or self.s.det_weights.is_file(),
                         "sha256_12": sha(self.s.det_weights), "imgsz": self.s.imgsz, "classes": list(CLASS_DISPLAY)},
            "classifier": {"file": str(self.s.cls_weights), "available": self.has_classifier(), "sha256_12": sha(self.s.cls_weights)},
            "device": self.device + (" (đã chuyển sang CPU do hết bộ nhớ GPU)" if self.fell_back_to_cpu else ""),
            "versions": versions,
        }

    # ---- 1. detector ----------------------------------------------------------------------------------------------
    def _yolo_detections(self, bgr: np.ndarray, conf: float, imgsz: int, iou: float) -> list[Detection]:
        r = self._predict(self.detector, bgr, conf=conf, imgsz=imgsz, iou=iou)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = _np(r.boxes.xyxy).reshape(-1, 4)
        score = _np(r.boxes.conf).reshape(-1)
        cls = _np(r.boxes.cls).astype(int).reshape(-1)
        names = {int(k): v for k, v in dict(self.detector.names).items()}
        dets = [Detection(names[int(c)], float(s), tuple(float(v) for v in b)) for b, s, c in zip(xyxy, score, cls)]
        return sorted(dets, key=lambda d: -d.score)

    def detect(self, image_path, conf: float | None = None, imgsz: int | None = None, iou: float | None = None) -> Analysis:
        conf = float(np.clip(self.s.default_conf if conf is None else conf, 0.01, 0.99))
        imgsz = int(imgsz or self.s.imgsz)
        iou = float(np.clip(self.s.iou_nms if iou is None else iou, 0.1, 0.95))
        t0 = time.perf_counter()
        bgr = load_image(image_path, self.s.max_upload_mb, self.s.min_side_px)
        t1 = time.perf_counter()
        dets = self._yolo_detections(bgr, conf, imgsz, iou)
        t2 = time.perf_counter()
        items = [(d.box, f"{d.label} {d.score:.2f}", CLASS_RGB.get(d.label, (255, 0, 0))) for d in dets]
        a = Analysis("yolo", render(bgr, items, self.s.max_display_side), dets, Path(str(image_path)).name, bgr.shape[:2],
                     {"conf": conf, "imgsz": imgsz, "iou_nms": iou, "device": self.device},
                     {"decode": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "total": (time.perf_counter() - t0) * 1e3},
                     crops=evidence_crops(bgr, dets))
        self._finish(a)
        return a

    # ---- 2. template (XOR) route ----------------------------------------------------------------------------------
    def _classify(self, bgr: np.ndarray, boxes: np.ndarray) -> list[tuple[str, float]]:
        if len(boxes) == 0:
            return []
        _, crop_utils = _pipeline()
        crops = [crop_utils.crop_square(bgr, *b) for b in boxes]
        out = []
        for r in self._predict(self.classifier, crops, imgsz=crop_utils.CROP_SIZE):
            out.append((display_name(r.names[int(r.probs.top1)]), float(r.probs.top1conf)))
        return out

    def _xor_stage(self, defect_path, template_path):
        """Run alignment + XOR. Returns (result dict from pipeline.xor_detect, boxes in test-image coordinates)."""
        pipeline, _ = _pipeline()
        with staged_files(defect_path, template_path) as (d, t):
            try:
                r = pipeline.xor_detect(t, d)
            except (ValueError, FileNotFoundError, cv2.error) as exc:
                log.info("alignment failed: %s", exc)
                raise AlignmentError(
                    "Không căn chỉnh được ảnh kiểm tra với ảnh mẫu. Kiểm tra: (1) ảnh mẫu là đúng bo mạch này, (2) hai ảnh chụp "
                    "cùng mặt, góc nhìn lệch không quá lớn, (3) ảnh đủ nét. Chi tiết kỹ thuật: " + str(exc)) from exc
        H = np.asarray(r["H"], dtype=np.float64)
        scale = abs(np.linalg.det(H[:2, :2])) if np.isfinite(H).all() else 0.0
        if not 0.25 <= scale <= 4.0:
            raise AlignmentError("Phép căn chỉnh cho kết quả bất thường (tỉ lệ co giãn quá lớn). Nhiều khả năng hai ảnh không phải "
                                 "cùng một bo mạch.")
        boxes = pipeline.to_defect_frame(r["new"], r["H"])
        h, w = r["defect"].shape[:2]
        if len(boxes):
            boxes = np.clip(boxes, 0, [w, h, w, h]).astype(np.float32)
        return r, boxes

    def xor_inspect(self, defect_path, template_path) -> Analysis:
        if template_path in (None, ""):
            raise InputError("Chế độ này cần thêm ảnh mẫu không lỗi của đúng bo mạch.")
        t0 = time.perf_counter()
        defect = load_image(defect_path, self.s.max_upload_mb, self.s.min_side_px)
        load_image(template_path, self.s.max_upload_mb, self.s.min_side_px)  # validate only
        _ = self.classifier  # fail early with a clear message if weights are missing
        t1 = time.perf_counter()
        r, boxes = self._xor_stage(defect_path, template_path)
        t2 = time.perf_counter()
        preds = self._classify(r["defect"], boxes)
        t3 = time.perf_counter()
        dets = sorted((Detection(n, p, tuple(float(v) for v in b), "xor") for b, (n, p) in zip(boxes, preds)), key=lambda d: -d.score)
        items = [(d.box, f"{d.label} {d.score:.2f}", CLASS_RGB.get(d.label, (255, 0, 0))) for d in dets]
        a = Analysis("xor", render(r["defect"], items, self.s.max_display_side), dets, Path(str(defect_path)).name, defect.shape[:2],
                     {"template": Path(str(template_path)).name, "xor_params": _pipeline()[0].XOR_PARAMS, "device": self.device},
                     {"decode": (t1 - t0) * 1e3, "align_xor": (t2 - t1) * 1e3, "classify": (t3 - t2) * 1e3,
                      "total": (time.perf_counter() - t0) * 1e3},
                     extra_images={"xor_map": self._xor_map(r)}, crops=evidence_crops(r["defect"], dets),
                     score_kind="classifier")
        if len(dets) > 40:
            a.notes.append("Có rất nhiều vùng khác biệt: nhiều khả năng căn chỉnh chưa đúng hoặc ảnh mẫu không cùng bo mạch.")
        a.notes.append("Độ tin cậy trong chế độ này là của bộ phân loại (mọi vùng khác biệt đều được gán một trong 6 lớp), "
                       "không phải xác suất vùng đó là lỗi thật.")
        self._finish(a)
        return a

    def self_check(self, canary_dir: Path | None = None, min_accuracy: float = 0.8) -> dict:
        """Classify the bundled reference crops (demo/assets/canary/<class>_<k>.png) and compare with the file names.

        Catches a crop classifier whose class names are permuted (unpatched best_cls.pt), which a names check cannot see.
        Returns {"checked": n, "correct": m, "ok": bool|None}; ok is None when there is nothing to check."""
        canary_dir = canary_dir or (REPO / "demo" / "assets" / "canary")
        files = sorted(p for p in canary_dir.glob("*.png")) if canary_dir.is_dir() else []
        truth, crops = [], []
        for p in files:
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                truth.append(display_name(p.stem.rsplit("_", 1)[0]))
                crops.append(img)
        if not crops or not self.has_classifier():
            return {"checked": 0, "correct": 0, "ok": None}
        _, crop_utils = _pipeline()
        preds = [display_name(r.names[int(r.probs.top1)]) for r in self._predict(self.classifier, crops, imgsz=crop_utils.CROP_SIZE)]
        correct = sum(p == t for p, t in zip(preds, truth))
        return {"checked": len(crops), "correct": correct, "ok": correct / len(crops) >= min_accuracy}

    def _xor_map(self, r) -> np.ndarray:
        base = r["aligned"].copy()
        mask = cv2.bitwise_and(r["xor_raw"], r["valid"]) > 0
        base[mask] = (0.35 * base[mask] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        return render(base, [], self.s.max_display_side)

    # ---- 3. cross-check -------------------------------------------------------------------------------------------
    def cross_check(self, defect_path, template_path, conf: float | None = None, match_iou: float = 0.1) -> Analysis:
        if template_path in (None, ""):
            raise InputError("Chế độ này cần thêm ảnh mẫu không lỗi của đúng bo mạch.")
        conf = float(np.clip(self.s.default_conf if conf is None else conf, 0.01, 0.99))
        t0 = time.perf_counter()
        defect = load_image(defect_path, self.s.max_upload_mb, self.s.min_side_px)
        load_image(template_path, self.s.max_upload_mb, self.s.min_side_px)
        _ = self.classifier
        yolo = self._yolo_detections(defect, conf, self.s.imgsz, self.s.iou_nms)
        t1 = time.perf_counter()
        r, xboxes = self._xor_stage(defect_path, template_path)
        xpreds = self._classify(r["defect"], xboxes)
        t2 = time.perf_counter()
        pairs, only_y, only_x = match_boxes([d.box for d in yolo], [tuple(b) for b in xboxes], match_iou)
        dets: list[Detection] = []
        for i, j in pairs:
            y, (xn, xp) = yolo[i], xpreds[j]
            note = "hai hướng cùng lớp" if xn == y.label else f"YOLO: {y.label}; phân loại XOR: {xn} ({xp:.2f})"
            dets.append(Detection(y.label, y.score, y.box, "both", note))
        dets += [Detection(yolo[i].label, yolo[i].score, yolo[i].box, "yolo") for i in only_y]
        dets += [Detection(xpreds[j][0], xpreds[j][1], tuple(float(v) for v in xboxes[j]), "xor") for j in only_x]
        items = [(d.box, f"{d.label} {d.score:.2f}", SOURCE_RGB[d.source]) for d in dets]
        a = Analysis("cross", render(r["defect"], items, self.s.max_display_side), dets, Path(str(defect_path)).name,
                     defect.shape[:2], {"conf": conf, "match_iou": match_iou, "device": self.device},
                     {"yolo": (t1 - t0) * 1e3, "xor_classify": (t2 - t1) * 1e3, "total": (time.perf_counter() - t0) * 1e3},
                     extra_images={"xor_map": self._xor_map(r)}, crops=evidence_crops(r["defect"], dets),
                     score_kind="mixed: detector (YOLO) / classifier (XOR-only)")
        a.notes.append("Quy tắc đối chiếu (ghép box ở IoU ≥ %.1f) chỉ để minh họa. Phương pháp kết hợp đã được đánh giá là tab "
                       "«YOLO + lọc XOR» (xem tab Thông tin mô hình)." % match_iou)
        self._finish(a)
        return a

    # ---- 4. YOLO verified by XOR ----------------------------------------------------------------------------------
    def fusion_check(self, defect_path, template_path, conf: float | None = None) -> Analysis:
        """Keep a YOLO box only if the aligned XOR change mask has enough changed pixels inside it (tools/fusion.py).
        Fixed board features look identical to the template, so boxes on them are dropped; boxes whose region cannot be
        verified (near the black warp border) are kept."""
        if template_path in (None, ""):
            raise InputError("Chế độ này cần thêm ảnh mẫu không lỗi của đúng bo mạch.")
        conf = float(np.clip(self.s.default_conf if conf is None else conf, 0.01, 0.99))
        t0 = time.perf_counter()
        defect = load_image(defect_path, self.s.max_upload_mb, self.s.min_side_px)
        load_image(template_path, self.s.max_upload_mb, self.s.min_side_px)
        yolo = self._yolo_detections(defect, conf, self.s.imgsz, self.s.iou_nms)
        t1 = time.perf_counter()
        r, _ = self._xor_stage(defect_path, template_path)
        pipeline, fusion, p = _pipeline()[0], _fusion(), _fusion_params()
        boxes = np.asarray([d.box for d in yolo], np.float32).reshape(-1, 4)
        mask = fusion.change_mask(r["xor_raw"], r["valid"], pipeline.XOR_PARAMS["close"])
        n_px, v_frac = fusion.evidence(fusion.box_to_aligned(boxes, r["H"]), mask, r["valid"], p["pad"])
        keep = fusion.keep_flags(n_px, v_frac, p["min_px"], p["min_valid"])
        t2 = time.perf_counter()
        dets, items, removed = [], [], 0
        for d, k, n, v in zip(yolo, keep, n_px, v_frac):
            if not k:
                removed += 1
                items.append((d.box, f"loại · {d.label} {d.score:.2f}", (150, 150, 150)))
                continue
            if v < p["min_valid"]:
                det = Detection(d.label, d.score, d.box, "yolo", "không xác minh được với ảnh mẫu (sát viền sau căn chỉnh), giữ nguyên")
            else:
                det = Detection(d.label, d.score, d.box, "both", f"có {int(n)} pixel khác ảnh mẫu trong vùng này")
            dets.append(det)
            items.append((det.box, f"{det.label} {det.score:.2f}", CLASS_RGB.get(det.label, (255, 0, 0))))
        a = Analysis("fusion", render(r["defect"], items, self.s.max_display_side), dets, Path(str(defect_path)).name,
                     defect.shape[:2], {"conf": conf, "min_px": p["min_px"], "pad": p["pad"], "removed": removed, "device": self.device},
                     {"yolo": (t1 - t0) * 1e3, "xor_verify": (t2 - t1) * 1e3, "total": (time.perf_counter() - t0) * 1e3},
                     extra_images={"xor_map": self._xor_map(r)}, crops=evidence_crops(r["defect"], dets), score_kind="detector (YOLO)")
        if removed:
            a.notes.append(f"Đã loại {removed}/{len(yolo)} vùng YOLO không khác ảnh mẫu (vẽ màu xám): thường là chi tiết thiết kế "
                           "của bo bị nhầm thành lỗi.")
        a.notes.append("Quy tắc này được đánh giá trên tập test (xem tab Thông tin mô hình). Số đo dùng ngưỡng tin cậy theo từng lớp "
                       "(configs/thresholds.json), còn demo dùng một ngưỡng chung nên kết quả có thể khác.")
        self._finish(a)
        return a

    def _finish(self, a: Analysis) -> None:
        if self.fell_back_to_cpu:
            a.notes.append("GPU hết bộ nhớ nên phân tích đã chạy trên CPU (chậm hơn, kết quả không đổi đáng kể).")
        log.info("req=%s mode=%s n=%d total_ms=%.0f", a.request_id, a.mode, len(a.detections), a.timings_ms.get("total", 0.0))