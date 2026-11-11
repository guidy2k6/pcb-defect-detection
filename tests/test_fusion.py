"""Kiem thu tools/fusion.py bang du lieu tong hop (chi can numpy, OpenCV, pytest)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fusion import box_to_aligned, change_mask, evidence, keep_flags, match_cls, prf  # noqa: E402


def test_box_to_aligned_translation():
    H = np.array([[1, 0, 10], [0, 1, -5], [0, 0, 1]], float)
    out = box_to_aligned([[0, 20, 30, 50]], H)
    assert np.allclose(out, [[10, 15, 40, 45]])
    assert box_to_aligned([], H).shape == (0, 4)


def test_evidence_and_keep():
    mask = np.zeros((200, 200), np.uint8)
    mask[50:60, 50:60] = 255                      # mot vung khac biet that (100 px)
    valid = np.full((200, 200), 255, np.uint8)
    boxes = [[45, 45, 70, 70],                    # chua vung khac biet
             [120, 120, 150, 150]]                # vung nen giong het bo mau (nhu hotspot co dinh)
    n, v = evidence(boxes, mask, valid, pad=0)
    assert n[0] == 100 and n[1] == 0 and np.allclose(v, 1.0)
    assert keep_flags(n, v, min_px=20).tolist() == [True, False]
    assert keep_flags(n, v, min_px=0).tolist() == [True, True]   # min_px=0 -> khong loc gi (bang YOLO thuan)


def test_unverifiable_box_is_kept():
    mask = np.zeros((100, 100), np.uint8)
    valid = np.zeros((100, 100), np.uint8)        # toan bo la vien den do warp
    n, v = evidence([[10, 10, 40, 40]], mask, valid, pad=0)
    assert v[0] == 0.0 and keep_flags(n, v, min_px=50).tolist() == [True]


def test_box_outside_image_has_no_evidence():
    mask = np.full((50, 50), 255, np.uint8)
    n, v = evidence([[60, 60, 80, 80]], mask, mask, pad=0)
    assert n[0] == 0 and v[0] == 0.0


def test_change_mask_removes_speckle_keeps_blob():
    raw = np.zeros((120, 120), np.uint8)
    raw[60, 60] = 255                             # nhieu 1 pixel
    raw[20:50, 20:50] = 255                       # vung that
    valid = np.full_like(raw, 255)
    m = change_mask(raw, valid, close=5)
    assert m[60, 60] == 0 and m[30, 30] == 255


def test_match_cls_is_class_aware_and_one_to_one():
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    pr = [[0, 0, 10, 10], [50, 50, 60, 60], [50, 50, 60, 60]]
    # box thu 2 dung vi tri nhung sai lop; box thu 3 dung lop nhung nhan do da duoc ghep -> FP
    assert match_cls(gt, [0, 1], pr, [0, 2, 1]) == (2, 1, 0)
    assert match_cls(gt, [0, 1], [[0, 0, 10, 10], [50, 50, 60, 60]], [0, 2]) == (1, 1, 1)
    assert match_cls([], [], [[0, 0, 5, 5]], [0]) == (0, 1, 0)
    assert match_cls([[0, 0, 5, 5]], [0], [], []) == (0, 0, 1)


def test_prf():
    p, r, f = prf(8, 2, 2)
    assert (round(p, 3), round(r, 3), round(f, 3)) == (0.8, 0.8, 0.8)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)