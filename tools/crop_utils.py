import cv2

CROP_SIZE = 128  # kich thuoc anh cat dua vao bo phan loai (train_cls.py, inference.py, collect_crops.py dung chung)
CROP_MARGIN = 0.5  # le them quanh box: canh vuong = canh dai nhat cua box x (1 + le)


def class_folder(name):
    """'Mouse Bite' -> 'mouse_bite' (ten thu muc lop cua bo phan loai)."""
    return name.strip().lower().replace(" ", "_")


def crop_square(img, x1, y1, x2, y2, margin=CROP_MARGIN, size=CROP_SIZE):
    """Cat vung vuong quanh box (toa do pixel), them le, dich vao trong anh neu tran, roi resize ve size x size."""
    h, w = img.shape[:2]
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    side = int(round(min(max(side, 32), h, w)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(round(min(max(cx - side / 2, 0), w - side)))
    top = int(round(min(max(cy - side / 2, 0), h - side)))
    patch = img[top:top + side, left:left + side]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
