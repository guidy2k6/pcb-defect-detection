"""Duong hai giai doan theo bai bao: can chinh SIFT -> nhi phan -> XOR -> khoanh vung -> phan loai (YOLOv8n-cls).

Dung chung cho tools/eval_pipeline.py va demo/app.py. Can chinh va XOR lay lai tu tools/baseline_xor.py (process_pair).
So voi box goc cua baseline_xor, bien the `new` them ba buoc:
  1. loai vien den do warp (valid mask) khoi anh XOR;
  2. median 5 + dong (close) kernel lon de noi cac manh XOR cua cung mot loi thanh mot box;
  3. phong box quanh tam (grow): XOR chi bat phan pixel khac, con box nhan ve rong hon quanh loi.
Tham so XOR_PARAMS duoc chon bang `python tools/eval_pipeline.py --split val --sweep` (chi tren val).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_xor import process_pair  # noqa: E402
from crop_utils import CROP_SIZE, crop_square  # noqa: E402

XOR_PARAMS = {"close": 27, "min_area": 250, "grow": 2.0}  # chon bang --sweep tren val (F1@IoU0.5 = 0.618)


def valid_mask(defect_shape, H, base_shape, erode=41):
    """Vung ma anh loi sau warp co du lieu that (bo vien den), co lai `erode` px de tranh vien."""
    h, w = base_shape[:2]
    m = cv2.warpPerspective(np.full(defect_shape[:2], 255, np.uint8), H, (w, h))
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))


def components(xor_raw, valid, close):
    """Anh XOR tho -> (stats cac thanh phan lien thong (x,y,w,h,area), kich thuoc anh). Buoc ton thoi gian nhat."""
    m = cv2.bitwise_and(xor_raw, valid)
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return st[1:], m.shape


def boxes_from(stats, shape, min_area, grow):
    """Loc theo dien tich roi phong box quanh tam; tra ve mang (N,4) xyxy trong toa do anh mau da can chinh."""
    H, W = shape
    boxes = []
    for x, y, w, h, a in stats:
        if a >= min_area:
            cx, cy, gw, gh = x + w / 2, y + h / 2, w * grow / 2, h * grow / 2
            boxes.append((max(cx - gw, 0), max(cy - gh, 0), min(cx + gw, W), min(cy + gh, H)))
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def xor_detect(template, defect, close=None, min_area=None, grow=None):
    """Tra ve dict: orig (box cua baseline_xor), new (box da cai tien), aligned (anh loi mau da can chinh),
    H, defect_shape, xor_raw, valid (de quet tham so ma khong can chay lai SIFT)."""
    p = {**XOR_PARAMS, **{k: v for k, v in (("close", close), ("min_area", min_area), ("grow", grow)) if v is not None}}
    imgs, raw, H = process_pair(Path(template), Path(defect))
    base_shape = imgs["baseline_gray"].shape
    defect_shape = imgs["defect_gray"].shape
    orig = np.asarray([(x, y, x + w, y + h) for x, y, w, h, _ in raw], np.float32).reshape(-1, 4)
    valid = valid_mask(defect_shape, H, base_shape)
    st, shape = components(imgs["xor_raw"], valid, p["close"])
    color = cv2.imread(str(defect), cv2.IMREAD_COLOR)
    aligned = cv2.warpPerspective(color, H, (base_shape[1], base_shape[0]))
    return {"orig": orig, "new": boxes_from(st, shape, p["min_area"], p["grow"]), "aligned": aligned, "defect": color, "H": H,
            "defect_shape": defect_shape, "xor_raw": imgs["xor_raw"], "valid": valid}


def to_defect_frame(boxes, H):
    """Box trong toa do anh mau (da can chinh) -> box trong anh loi goc (bien doi nghich H, lay hop bao)."""
    if len(boxes) == 0:
        return boxes
    pts = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], 1).reshape(-1, 1, 2)
    t = cv2.perspectiveTransform(pts.astype(np.float32), np.linalg.inv(H)).reshape(-1, 4, 2)
    return np.column_stack([t[:, :, 0].min(1), t[:, :, 1].min(1), t[:, :, 0].max(1), t[:, :, 1].max(1)]).astype(np.float32)


def classify_boxes(cls_model, r, boxes, crop_from="original"):
    """Phan loai cac box cua ket qua xor_detect `r`. crop_from='original' (mac dinh): cat tu anh loi goc, khong bi lam mo do
    noi suy khi warp, giong luc train; 'aligned': cat tu anh da can chinh (de doi chieu)."""
    if crop_from == "original":
        return classify(cls_model, r["defect"], to_defect_frame(boxes, r["H"]))
    return classify(cls_model, r["aligned"], boxes)


def classify(cls_model, aligned, boxes):
    """Cat vung vuong quanh moi box (giong luc train) roi phan loai. Tra ve list (ten_lop, do_tin_cay)."""
    if len(boxes) == 0:
        return []
    crops = [crop_square(aligned, *b) for b in boxes]
    out = []
    for r in cls_model.predict(crops, imgsz=CROP_SIZE, verbose=False):
        out.append((r.names[int(r.probs.top1)], float(r.probs.top1conf)))
    return out


def draw(img, boxes, labels=None, color=(0, 0, 255), scale=1.0):
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, max(2, int(3 / scale)))
        if labels:
            cv2.putText(out, labels[i], (int(x1), max(int(y1) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9 / scale, color, max(2, int(2 / scale)), cv2.LINE_AA)
    return out
