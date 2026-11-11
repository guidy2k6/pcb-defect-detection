"""Ket hop YOLO + XOR: dung ban do khac biet XOR de xac minh box YOLO (ham thuan, chi can numpy va OpenCV).

Y tuong: chi tiet thiet ke co dinh cua bo (vong tron in, cho noi duong mach...) giong het anh bo mau, nen XOR
khong thay khac biet o do. Box YOLO nam o vung do vi vay khong co "bang chung thay doi" -> bi loai.
Loi that thi lam anh khac bo mau -> co pixel XOR trong box -> duoc giu.

Quy tac (mot tham so chinh `min_px`, chon tren val bang tools/eval_fusion.py --sweep):
    giu box  <=>  so pixel khac biet trong box (mo rong `pad` px) >= min_px
              HOAC phan vung hop le (khong phai vien den do warp) cua box < min_valid (khong xac minh duoc thi giu).
"""
import cv2
import numpy as np


def box_to_aligned(boxes, H):
    """Box xyxy trong anh loi goc -> hop bao trong toa do anh mau da can chinh (H anh xa anh loi -> anh mau)."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes
    pts = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], 1).reshape(-1, 1, 2)
    t = cv2.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 4, 2)
    return np.column_stack([t[:, :, 0].min(1), t[:, :, 1].min(1), t[:, :, 0].max(1), t[:, :, 1].max(1)]).astype(np.float32)


def change_mask(xor_raw, valid, close):
    """Anh XOR tho -> mat na khac biet da loc nhieu (cung cac buoc dau cua pipeline.components, nhung giu lai mat na)."""
    m = cv2.bitwise_and(xor_raw, valid)
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def evidence(boxes_al, mask, valid, pad=8):
    """Voi moi box (toa do anh mau): (so pixel khac biet, ti le vung hop le) trong box mo rong `pad` px."""
    H, W = mask.shape[:2]
    n_px, v_frac = [], []
    for x1, y1, x2, y2 in np.asarray(boxes_al, np.float32).reshape(-1, 4):
        a, b = int(max(np.floor(x1) - pad, 0)), int(max(np.floor(y1) - pad, 0))
        c, d = int(min(np.ceil(x2) + pad, W)), int(min(np.ceil(y2) + pad, H))
        if c <= a or d <= b:  # box nam ngoai anh mau: khong co bang chung
            n_px.append(0); v_frac.append(0.0)
            continue
        n_px.append(int(np.count_nonzero(mask[b:d, a:c])))
        v_frac.append(float(np.count_nonzero(valid[b:d, a:c])) / ((c - a) * (d - b)))
    return np.asarray(n_px, int), np.asarray(v_frac, float)


def keep_flags(n_px, v_frac, min_px, min_valid=0.5):
    """True = giu box. Khong xac minh duoc (vung hop le qua it) thi giu, de khong mat box dung vi can chinh/vien."""
    return (np.asarray(n_px) >= min_px) | (np.asarray(v_frac) < min_valid)


def _iou(a, b):
    ix1, iy1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    ix2, iy2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_cls(gt, gt_cls, pr, pr_cls, thr=0.5):
    """Ghep tham lam theo IoU giam dan, moi nhan/box dung toi da mot lan, CHI ghep cung lop.
    Tra ve (tp, fp, fn). Box nham lop tinh la FP va nhan do tinh la FN (khat khe hon cach tinh 'nham lop' rieng)."""
    gt, pr = np.asarray(gt, np.float32).reshape(-1, 4), np.asarray(pr, np.float32).reshape(-1, 4)
    if len(gt) == 0 or len(pr) == 0:
        return 0, len(pr), len(gt)
    iou = _iou(gt, pr)
    iou[np.asarray(gt_cls)[:, None] != np.asarray(pr_cls)[None, :]] = 0.0
    used_g, used_p, tp = set(), set(), 0
    for flat in np.argsort(-iou, axis=None):
        g, p = np.unravel_index(flat, iou.shape)
        if iou[g, p] < thr:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p); tp += 1
    return tp, len(pr) - tp, len(gt) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)