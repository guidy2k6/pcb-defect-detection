"""Tao nhieu (xoay, nghieng, mo, toi) cho anh bo mach va bien doi box theo anh.

Chi dung numpy + OpenCV, khong ngau nhien (cung dau vao -> cung dau ra) de so lieu tai lap duoc.
Box dung dinh dang xyxy theo pixel cua anh goc: mang (N, 4) float.

Quy uoc:
  - Xoay boi 90 do: chinh xac tung pixel, box chinh xac.
  - Xoay goc le / nghieng: toa do 4 goc cua box duoc bien doi roi lay hop bao. Voi box nho,
    hop bao no ra (45 do: dien tich gap >= 2 lan), nen nhan sau xoay khong con khit. `tightness`
    do viec nay: dien tich box goc / dien tich hop bao (1.0 = khit). IoU toi da giua box tot nhat
    va nhan sau bien doi xap xi bang `tightness`, nen so mAP50 o muc xoay goc le la so thap hon thuc.
  - Mo / toi: giu nguyen box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class Perturbation:
    name: str      # ten muc, vd "rot15"
    family: str    # nhom: clean | rot | tilt | blur | motion | dark
    level: float   # gia tri muc (do, sigma, ...)
    fn: Callable   # fn(img, boxes_xyxy) -> (img, boxes_xyxy)


def _clip_boxes(boxes: np.ndarray, w: int, h: int, min_side: float = 1.0) -> np.ndarray:
    """Cat box vao anh, bo box con nho hon `min_side` pixel."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    keep = ((b[:, 2] - b[:, 0]) >= min_side) & ((b[:, 3] - b[:, 1]) >= min_side)
    return b, keep


def _envelope(corners: np.ndarray) -> np.ndarray:
    """corners: (N, 4, 2) -> hop bao xyxy (N, 4)."""
    return np.stack([corners[..., 0].min(1), corners[..., 1].min(1),
                     corners[..., 0].max(1), corners[..., 1].max(1)], axis=1)


def _box_corners(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1)


def _warp_boxes_affine(boxes: np.ndarray, M: np.ndarray) -> np.ndarray:
    c = _box_corners(boxes)
    ones = np.ones((c.shape[0], 4, 1))
    return _envelope(np.concatenate([c, ones], axis=2) @ M.T)


def rotate90(img: np.ndarray, boxes: np.ndarray, k: int):
    """Xoay nguoc chieu kim dong ho k*90 do (khong noi suy). Tra ve (anh, box)."""
    k %= 4
    h, w = img.shape[:2]
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if k == 0:
        return img.copy(), b.copy()
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    if k == 1:   # nguoc chieu kim dong ho 90
        out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        nb = np.stack([y1, w - x2, y2, w - x1], axis=1)
    elif k == 2:
        out = cv2.rotate(img, cv2.ROTATE_180)
        nb = np.stack([w - x2, h - y2, w - x1, h - y1], axis=1)
    else:        # 270
        out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        nb = np.stack([h - y2, x1, h - y1, x2], axis=1)
    return out, nb


def rotate_expand(img: np.ndarray, boxes: np.ndarray, angle_deg: float):
    """Xoay goc bat ky quanh tam, mo rong khung de khong mat noi dung. Vung trong = den."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    nb, keep = _clip_boxes(_warp_boxes_affine(boxes, M), nw, nh)
    return out, nb, keep


def tilt(img: np.ndarray, boxes: np.ndarray, strength: float):
    """Mo phong nghieng (perspective): canh tren thu hep `strength` * be rong moi ben.

    Khung anh giu nguyen kich thuoc; vung ngoai noi dung la den.
    """
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = strength * w
    dst = np.float32([[d, 0], [w - d, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    c = _box_corners(boxes).astype(np.float32)
    if len(c) == 0:
        return out, np.zeros((0, 4)), np.zeros(0, dtype=bool)
    warped = cv2.perspectiveTransform(c.reshape(-1, 1, 2), H).reshape(-1, 4, 2)
    nb, keep = _clip_boxes(_envelope(warped.astype(np.float64)), w, h)
    return out, nb, keep


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def motion_blur(img: np.ndarray, length: int, angle_deg: float = 0.0) -> np.ndarray:
    """Nhoe do chuyen dong: nhan duong thang dai `length` pixel, huong `angle_deg`."""
    length = max(int(length), 1)
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    k = k / s if s > 0 else np.ones((1, 1), np.float32)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)


def darken(img: np.ndarray, factor: float) -> np.ndarray:
    """Giam sang: nhan moi kenh voi `factor` (0.5 = toi di mot nua)."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(img.dtype)


def tightness(orig_boxes: np.ndarray, new_boxes: np.ndarray) -> float:
    """Trung binh (dien tich box goc / dien tich hop bao moi). 1.0 = nhan van khit."""
    o = np.asarray(orig_boxes, dtype=np.float64).reshape(-1, 4)
    n = np.asarray(new_boxes, dtype=np.float64).reshape(-1, 4)
    if len(o) == 0 or len(o) != len(n):
        return float("nan")
    ao = (o[:, 2] - o[:, 0]) * (o[:, 3] - o[:, 1])
    an = (n[:, 2] - n[:, 0]) * (n[:, 3] - n[:, 1])
    return float(np.mean(np.minimum(ao / np.maximum(an, 1e-9), 1.0)))


def _wrap_rot90(k):
    return lambda img, b: (*rotate90(img, b, k), np.ones(len(np.reshape(b, (-1, 4))), dtype=bool))


def _wrap_rot(a):
    return lambda img, b: rotate_expand(img, b, a)


def _wrap_tilt(s):
    return lambda img, b: tilt(img, b, s)


def _wrap_img(f):
    """Phep chi doi anh, giu nguyen box."""
    def g(img, b):
        b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
        return f(img), b.copy(), np.ones(len(b), dtype=bool)
    return g


PERTURBATIONS: dict[str, Perturbation] = {p.name: p for p in [
    Perturbation("clean", "clean", 0, _wrap_img(lambda im: im.copy())),
    Perturbation("rot15", "rot", 15, _wrap_rot(15)),
    Perturbation("rot45", "rot", 45, _wrap_rot(45)),
    Perturbation("rot90", "rot", 90, _wrap_rot90(1)),
    Perturbation("rot180", "rot", 180, _wrap_rot90(2)),
    Perturbation("tilt5", "tilt", 0.05, _wrap_tilt(0.05)),
    Perturbation("tilt10", "tilt", 0.10, _wrap_tilt(0.10)),
    Perturbation("tilt15", "tilt", 0.15, _wrap_tilt(0.15)),
    Perturbation("blur_s1", "blur", 1, _wrap_img(lambda im: gaussian_blur(im, 1))),
    Perturbation("blur_s2", "blur", 2, _wrap_img(lambda im: gaussian_blur(im, 2))),
    Perturbation("blur_s3", "blur", 3, _wrap_img(lambda im: gaussian_blur(im, 3))),
    Perturbation("blur_s5", "blur", 5, _wrap_img(lambda im: gaussian_blur(im, 5))),
    Perturbation("motion15", "motion", 15, _wrap_img(lambda im: motion_blur(im, 15, 0.0))),
    Perturbation("dark50", "dark", 0.5, _wrap_img(lambda im: darken(im, 0.5))),
]}


def apply(name: str, img: np.ndarray, boxes_xyxy: np.ndarray):
    """Ap dung muc nhieu `name`. Tra ve (anh moi, box moi xyxy, tightness).

    Box ra ngoai anh sau khi bien doi bi cat; box con nho hon 1 pixel bi bo.
    """
    p = PERTURBATIONS[name]
    out, nb, keep = p.fn(img, boxes_xyxy)
    nb = np.asarray(nb, dtype=np.float64).reshape(-1, 4)
    keep = np.asarray(keep, dtype=bool)
    orig = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    return out, nb[keep], tightness(orig[keep], nb[keep]), keep


def yolo_to_xyxy(rows, w: int, h: int):
    """rows: [(cls, xc, yc, bw, bh)] chuan hoa -> (cls (N,), xyxy (N,4) pixel)."""
    if not rows:
        return np.zeros(0, dtype=int), np.zeros((0, 4))
    a = np.asarray(rows, dtype=np.float64)
    cls = a[:, 0].astype(int)
    xc, yc, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return cls, np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)


def xyxy_to_yolo(cls, boxes, w: int, h: int):
    """Nguoc lai: tra ve list (cls, xc, yc, bw, bh) chuan hoa theo kich thuoc anh moi (w, h)."""
    out = []
    for c, (x1, y1, x2, y2) in zip(cls, np.asarray(boxes).reshape(-1, 4)):
        out.append((int(c), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return out
