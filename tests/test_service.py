"""Unit tests for demo/service.py. They need only numpy, OpenCV and pytest: models are replaced by small fakes."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from demo.config import DETECTOR_CLASSES, STALE_DETECTOR_CLASSES, Settings
from demo.service import (AlignmentError, InputError, ModelError, PCBInspector, check_classifier_names, check_detector_names,
                          detections_table, display_name, load_image, match_boxes, to_csv, to_json, write_exports)


# ---------------------------------------------------------------------------------------------------------------- fakes
class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.array(xyxy, float).reshape(-1, 4), np.array(conf, float), np.array(cls, float)

    def __len__(self):
        return len(self.conf)


class FakeDetector:
    names = DETECTOR_CLASSES

    def __init__(self, xyxy=(), conf=(), cls=(), fail_on_gpu=False):
        self.out = FakeBoxes(xyxy, conf, cls)
        self.fail_on_gpu = fail_on_gpu
        self.devices = []

    def predict(self, source, device="cpu", **kwargs):
        self.devices.append(device)
        if self.fail_on_gpu and device != "cpu":
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.88 GiB")
        return [SimpleNamespace(boxes=self.out)]


class FakeClassifier:
    names = {0: "open_circuit", 1: "missing_hole", 2: "spur", 3: "short", 4: "mouse_bite", 5: "spurious_copper"}

    def predict(self, crops, imgsz=128, device="cpu", **kwargs):
        return [SimpleNamespace(names=self.names, probs=SimpleNamespace(top1=2, top1conf=0.9)) for _ in crops]


def settings(tmp_path):
    return Settings(device="cpu", det_weights=tmp_path / "none.pt", cls_weights=tmp_path / "none_cls.pt", results_dir=tmp_path,
                    examples_dir=tmp_path / "examples")


def write_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))
    return path


@pytest.fixture
def plain_image(tmp_path):
    img = np.full((600, 800, 3), (30, 120, 40), np.uint8)
    cv2.circle(img, (400, 300), 20, (200, 200, 200), -1)
    return write_png(tmp_path / "board.png", img)


# ----------------------------------------------------------------------------------------------------- input validation
def test_load_image_accepts_valid(plain_image):
    assert load_image(plain_image).shape == (600, 800, 3)


@pytest.mark.parametrize("bad", [None, "", "/nonexistent/file.png"])
def test_load_image_rejects_missing(bad):
    with pytest.raises(InputError):
        load_image(bad)


def test_load_image_rejects_wrong_type_corrupt_small_and_large(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(InputError, match="Định dạng"):
        load_image(txt)
    corrupt = tmp_path / "c.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(InputError, match="Không đọc được"):
        load_image(corrupt)
    tiny = write_png(tmp_path / "t.png", np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(InputError, match="quá nhỏ"):
        load_image(tiny)
    big = write_png(tmp_path / "b.png", np.random.randint(0, 255, (600, 800, 3), np.uint8))
    with pytest.raises(InputError, match="vượt giới hạn"):
        load_image(big, max_mb=0)


def test_load_image_handles_non_ascii_path(tmp_path):
    p = write_png(tmp_path / "bảng mạch lỗi.png", np.zeros((300, 300, 3), np.uint8))
    assert load_image(p).shape[:2] == (300, 300)


# -------------------------------------------------------------------------------------------------------------- helpers
def test_display_name():
    assert display_name("open_circuit") == "Open Circuit"
    assert display_name("Spurious Copper") == "Spurious Copper"


def test_match_boxes_one_to_one():
    a = [(0, 0, 10, 10), (100, 100, 120, 120)]
    b = [(1, 1, 11, 11), (500, 500, 510, 510)]
    pairs, only_a, only_b = match_boxes(a, b, 0.3)
    assert pairs == [(0, 0)] and only_a == [1] and only_b == [1]


def test_detector_name_guard():
    check_detector_names(DETECTOR_CLASSES, "best.pt")
    with pytest.raises(ModelError, match="fix_class_names"):
        check_detector_names(STALE_DETECTOR_CLASSES, "best.pt")
    with pytest.raises(ModelError):
        check_detector_names({0: "a"}, "best.pt")


def test_classifier_name_guard():
    check_classifier_names(FakeClassifier.names)
    with pytest.raises(ModelError):
        check_classifier_names({0: "foo"})


# ----------------------------------------------------------------------------------------------------------- detector
def test_detect_sorts_by_confidence_and_exports(tmp_path, plain_image):
    det = FakeDetector([(380, 280, 420, 320), (50, 50, 90, 80)], [0.45, 0.8], [3, 1])
    a = PCBInspector(settings(tmp_path), detector=det).detect(plain_image, conf=0.3)
    assert [d.label for d in a.detections] == ["Mouse Bite", "Short"]  # sorted by score
    assert a.counts()["Short"] == 1 and a.counts()["Spur"] == 0
    assert a.annotated.ndim == 3 and a.annotated.shape[2] == 3
    assert len(a.crops) == 2
    csv_text = to_csv(a)
    assert csv_text.splitlines()[0].startswith("id,label,score") and "Mouse Bite" in csv_text
    assert '"counts"' in to_json(a, {"device": "cpu"})
    assert len(detections_table(a)) == 2
    csv_path, json_path = write_exports(a, {"device": "cpu"})
    assert open(csv_path, encoding="utf-8-sig", newline="").read() == csv_text
    assert b"\r" not in open(csv_path, "rb").read()


def test_detect_empty_result(tmp_path, plain_image):
    a = PCBInspector(settings(tmp_path), detector=FakeDetector()).detect(plain_image)
    assert a.detections == [] and sum(a.counts().values()) == 0


def test_conf_is_clamped(tmp_path, plain_image):
    a = PCBInspector(settings(tmp_path), detector=FakeDetector()).detect(plain_image, conf=5)
    assert a.params["conf"] == 0.99


def test_gpu_oom_falls_back_to_cpu_once(tmp_path, plain_image):
    det = FakeDetector([(380, 280, 420, 320)], [0.7], [3], fail_on_gpu=True)
    insp = PCBInspector(settings(tmp_path), detector=det)
    insp.device = "0"
    a = insp.detect(plain_image)
    assert insp.device == "cpu" and insp.fell_back_to_cpu
    assert det.devices == ["0", "cpu"] and len(a.detections) == 1
    assert any("CPU" in n for n in a.notes)
    insp.detect(plain_image)
    assert det.devices[-1] == "cpu" and det.devices.count("0") == 1  # does not retry the GPU


def test_non_oom_errors_propagate(tmp_path, plain_image):
    class Broken(FakeDetector):
        def predict(self, *a, **k):
            raise RuntimeError("something else")
    insp = PCBInspector(settings(tmp_path), detector=Broken())
    insp.device = "0"
    with pytest.raises(RuntimeError, match="something else"):
        insp.detect(plain_image)


def test_missing_weights_message(tmp_path, plain_image):
    with pytest.raises(ModelError, match="models/"):
        PCBInspector(settings(tmp_path)).detect(plain_image)


def test_template_mode_requires_template_and_classifier(tmp_path, plain_image):
    insp = PCBInspector(settings(tmp_path), detector=FakeDetector())
    with pytest.raises(InputError):
        insp.xor_inspect(plain_image, None)
    with pytest.raises(ModelError):
        insp.xor_inspect(plain_image, plain_image)


# --------------------------------------------------------------------------------------- template (XOR) route, synthetic
def synthetic_pair(tmp_path, seed=0):
    """A feature-rich fake board, plus the same board rotated/shifted with one defect (a missing-copper gap) drawn on it."""
    rng = np.random.default_rng(seed)
    board = np.full((1000, 1400, 3), (40, 110, 50), np.uint8)
    for _ in range(140):  # traces and pads give SIFT plenty of corners
        x, y = int(rng.integers(30, 1360)), int(rng.integers(30, 960))
        if rng.random() < 0.5:
            cv2.line(board, (x, y), (x + int(rng.integers(-150, 150)), y + int(rng.integers(-150, 150))), (90, 190, 110), 8)
        else:
            cv2.circle(board, (x, y), int(rng.integers(8, 20)), (190, 190, 200), -1)
    board = cv2.GaussianBlur(board, (3, 3), 0)
    template = board.copy()
    defect = board.copy()
    cv2.circle(defect, (700, 500), 26, (15, 40, 20), -1)  # the injected defect
    M = cv2.getRotationMatrix2D((700, 500), 1.5, 1.0)
    M[:, 2] += (12, -9)
    warped = cv2.warpAffine(defect, M, (1400, 1000), borderMode=cv2.BORDER_REPLICATE)
    centre_in_warped = (M @ np.array([700, 500, 1.0])).tolist()
    return write_png(tmp_path / "template.png", template), write_png(tmp_path / "defect.png", warped), centre_in_warped


def test_xor_finds_injected_defect(tmp_path):
    tpl, dfc, (cx, cy) = synthetic_pair(tmp_path)
    insp = PCBInspector(settings(tmp_path), detector=FakeDetector(), classifier=FakeClassifier())
    a = insp.xor_inspect(dfc, tpl)
    assert a.score_kind == "classifier" and "xor_map" in a.extra_images
    hit = [d for d in a.detections if d.box[0] - 30 <= cx <= d.box[2] + 30 and d.box[1] - 30 <= cy <= d.box[3] + 30]
    assert hit, f"no box near the injected defect at ({cx:.0f},{cy:.0f}); got {[d.box for d in a.detections]}"
    assert hit[0].label == "Spur" and hit[0].source == "xor"  # the fake classifier always answers class 2 (spur)


def test_cross_check_marks_agreement(tmp_path):
    tpl, dfc, (cx, cy) = synthetic_pair(tmp_path)
    det = FakeDetector([(cx - 30, cy - 30, cx + 30, cy + 30), (50, 50, 120, 120)], [0.8, 0.5], [4, 1])
    a = PCBInspector(settings(tmp_path), detector=det, classifier=FakeClassifier()).cross_check(dfc, tpl)
    sources = {d.source for d in a.detections}
    assert "both" in sources and "yolo" in sources  # the real defect is seen by both; the box at (50,50) only by YOLO
    both = [d for d in a.detections if d.source == "both"][0]
    assert both.label == "Spur" and "cùng lớp" in both.note


def test_fusion_check_drops_unchanged_region_keeps_defect(tmp_path):
    tpl, dfc, (cx, cy) = synthetic_pair(tmp_path)
    det = FakeDetector([(cx - 30, cy - 30, cx + 30, cy + 30), (50, 50, 120, 120)], [0.8, 0.5], [4, 1])
    a = PCBInspector(settings(tmp_path), detector=det, classifier=FakeClassifier()).fusion_check(dfc, tpl)
    assert a.mode == "fusion" and a.params["removed"] == 1
    assert [d.label for d in a.detections] == ["Spur"]  # the box on the unchanged region is dropped, the injected defect kept
    assert "pixel khác ảnh mẫu" in a.detections[0].note


def test_fusion_check_requires_template(tmp_path, plain_image):
    insp = PCBInspector(settings(tmp_path), detector=FakeDetector(), classifier=FakeClassifier())
    with pytest.raises(InputError):
        insp.fusion_check(plain_image, None)


def test_xor_with_unrelated_template_raises_alignment_error(tmp_path):
    tpl, dfc, _ = synthetic_pair(tmp_path, seed=0)
    sub = tmp_path / "other"
    sub.mkdir()
    other, _, _ = synthetic_pair(sub, seed=99)
    insp = PCBInspector(settings(tmp_path), detector=FakeDetector(), classifier=FakeClassifier())
    with pytest.raises(AlignmentError):
        insp.xor_inspect(dfc, other)


# ------------------------------------------------------------------------------------------------------ classifier canary
class CodedClassifier:
    """Fake classifier that reads the class from the crop's top-left pixel value (class index * 10)."""
    names = {0: "missing_hole", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: "spur", 5: "spurious_copper"}

    def __init__(self, permute=False):
        self.permute = permute

    def predict(self, crops, imgsz=128, device="cpu", **kwargs):
        out = []
        for c in crops:
            k = int(c[0, 0, 0]) // 10
            k = (k + 1) % 6 if self.permute else k
            out.append(SimpleNamespace(names=self.names, probs=SimpleNamespace(top1=k, top1conf=0.9)))
        return out


def write_canary(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(CodedClassifier.names.values()):
        for j in (1, 2):
            write_png(folder / f"{name}_{j}.png", np.full((128, 128, 3), k * 10, np.uint8))


def test_self_check_passes_and_detects_permutation(tmp_path):
    write_canary(tmp_path / "canary")
    good = PCBInspector(settings(tmp_path), detector=FakeDetector(), classifier=CodedClassifier())
    assert good.self_check(tmp_path / "canary") == {"checked": 12, "correct": 12, "ok": True}
    bad = PCBInspector(settings(tmp_path), detector=FakeDetector(), classifier=CodedClassifier(permute=True))
    assert bad.self_check(tmp_path / "canary")["ok"] is False


def test_self_check_skips_without_canary(tmp_path):
    insp = PCBInspector(settings(tmp_path), detector=FakeDetector(), classifier=CodedClassifier())
    assert insp.self_check(tmp_path / "nothing")["ok"] is None