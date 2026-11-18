import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import perturb as P  # noqa: E402

W, H = 400, 300
BOX = np.array([[100.0, 60.0, 160.0, 120.0]])  # hinh chu nhat 60x60 sang tren nen toi


def scene():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    x1, y1, x2, y2 = BOX[0].astype(int)
    img[y1:y2, x1:x2] = 255
    return img


def bright_bbox(img, thr=128):
    ys, xs = np.where(img[..., 0] > thr)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


@pytest.mark.parametrize("name", ["rot90", "rot180"])
def test_rot90_boxes_are_exact(name):
    out, nb, tight, keep = P.apply(name, scene(), BOX)
    assert keep.all() and tight == pytest.approx(1.0)
    np.testing.assert_allclose(nb[0], bright_bbox(out), atol=1)


def test_rot90_changes_shape_and_k4_is_identity():
    out, nb, *_ = P.apply("rot90", scene(), BOX)
    assert out.shape[:2] == (W, H)
    img, b = scene(), BOX
    for _ in range(4):
        img, b = P.rotate90(img, b, 1)
    assert (img == scene()).all()
    np.testing.assert_allclose(b, BOX)


@pytest.mark.parametrize("name,min_iou", [("rot15", 0.7), ("rot45", 0.45), ("tilt5", 0.9), ("tilt10", 0.85), ("tilt15", 0.8)])
def test_box_envelops_rotated_or_tilted_content(name, min_iou):
    out, nb, tight, keep = P.apply(name, scene(), BOX)
    assert keep.all() and len(nb) == 1
    content = bright_bbox(out)
    # hop bao phai chua noi dung (cho sai so noi suy 2 px) va khong qua lon
    assert nb[0][0] <= content[0] + 2 and nb[0][1] <= content[1] + 2
    assert nb[0][2] >= content[2] - 2 and nb[0][3] >= content[3] - 2
    assert iou(nb[0], content) >= min_iou
    assert 0 < tight <= 1.0


def test_rot45_label_inflation_is_reported():
    _, _, tight, _ = P.apply("rot45", scene(), BOX)
    assert tight == pytest.approx(0.5, abs=0.02)  # hinh vuong xoay 45 do: hop bao gap doi dien tich


def test_rotate_expand_keeps_all_content_inside_canvas():
    out, nb, tight, keep = P.apply("rot15", scene(), BOX)
    assert out.shape[0] > H and out.shape[1] > W
    assert (nb >= 0).all() and (nb[:, [0, 2]] <= out.shape[1]).all() and (nb[:, [1, 3]] <= out.shape[0]).all()


@pytest.mark.parametrize("name", ["blur_s1", "blur_s2", "blur_s3", "blur_s5", "motion15", "dark50", "clean"])
def test_photometric_levels_keep_boxes(name):
    img = scene()
    out, nb, tight, keep = P.apply(name, img, BOX)
    assert out.shape == img.shape and keep.all()
    np.testing.assert_allclose(nb, BOX)
    assert tight == pytest.approx(1.0)


def test_blur_increases_with_sigma_and_dark_lowers_brightness():
    img = scene()
    edge = lambda im: float(np.abs(np.diff(im[..., 0].astype(float), axis=1)).max())  # noqa: E731
    e = [edge(P.apply(n, img, BOX)[0]) for n in ("clean", "blur_s1", "blur_s2", "blur_s3", "blur_s5")]
    assert all(a > b for a, b in zip(e, e[1:]))
    assert P.apply("dark50", img, BOX)[0].max() == 127
    assert (P.apply("clean", img, BOX)[0] == img).all()


def test_motion_blur_is_directional():
    img = scene()
    out = P.motion_blur(img, 15, 0.0)
    dx = np.abs(np.diff(out[..., 0].astype(float), axis=1)).max()
    dy = np.abs(np.diff(out[..., 0].astype(float), axis=0)).max()
    assert dx < dy  # lam mo theo phuong ngang: canh dung mem hon canh ngang


def test_boxes_leaving_the_frame_are_dropped_or_clipped():
    img = scene()
    far = np.array([[0.0, 0.0, 30.0, 30.0], [100.0, 60.0, 160.0, 120.0]])
    out, nb, tight, keep = P.apply("tilt15", img, far)
    assert len(nb) == int(keep.sum())
    assert (nb[:, [0, 2]] <= W).all() and (nb >= 0).all()


def test_yolo_roundtrip_and_empty_inputs():
    rows = [(2, 0.5, 0.5, 0.1, 0.2)]
    cls, xyxy = P.yolo_to_xyxy(rows, 1000, 500)
    back = P.xyxy_to_yolo(cls, xyxy, 1000, 500)
    assert back[0][0] == 2 and back[0][1:] == pytest.approx(rows[0][1:])
    assert P.yolo_to_xyxy([], 10, 10)[1].shape == (0, 4)
    for n in P.PERTURBATIONS:
        out, nb, t, keep = P.apply(n, scene(), np.zeros((0, 4)))
        assert nb.shape == (0, 4) and out.ndim == 3


def test_registry_has_the_levels_in_the_plan():
    need = {"rot15", "rot45", "rot90", "rot180", "blur_s1", "blur_s2", "blur_s3", "blur_s5", "motion15", "dark50"}
    assert need <= set(P.PERTURBATIONS)
    assert any(p.family == "tilt" for p in P.PERTURBATIONS.values())
