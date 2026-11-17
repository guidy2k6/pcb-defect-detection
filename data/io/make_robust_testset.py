"""Generate rotated, tilted, and blurred variants of the held-out YOLO test set.

    python data/io/make_robust_testset.py --data configs/data_board.yaml
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from yolo_utils import label_path, load_split, read_labels  # noqa: E402


def _transform_boxes(boxes, matrix, width, height):
    transformed = []
    for class_id, xc, yc, box_width, box_height in boxes:
        x1, x2 = (xc - box_width / 2) * width, (xc + box_width / 2) * width
        y1, y2 = (yc - box_height / 2) * height, (yc + box_height / 2) * height
        corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
            dtype=np.float32,
        )
        points = cv2.perspectiveTransform(corners, matrix)[0]
        x_min = float(np.clip(points[:, 0].min(), 0, width))
        x_max = float(np.clip(points[:, 0].max(), 0, width))
        y_min = float(np.clip(points[:, 1].min(), 0, height))
        y_max = float(np.clip(points[:, 1].max(), 0, height))
        if x_max > x_min and y_max > y_min:
            transformed.append((
                class_id,
                (x_min + x_max) / (2 * width),
                (y_min + y_max) / (2 * height),
                (x_max - x_min) / width,
                (y_max - y_min) / height,
            ))
    return transformed


def _rotation_matrix(width, height, angle):
    affine = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return np.vstack((affine, (0, 0, 1))).astype(np.float32)


def _tilt_matrix(width, height, fraction, far_edge):
    if far_edge == "top":
        destination = [[width * fraction, 0], [width * (1 - fraction), 0], [width, height], [0, height]]
    else:
        destination = [[0, 0], [width, 0], [width * (1 - fraction), height], [width * fraction, height]]
    source = [[0, 0], [width, 0], [width, height], [0, height]]
    return cv2.getPerspectiveTransform(np.float32(source), np.float32(destination))


def _read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read test image: {path}")
    return image


def _write_variant(output_root, image_path, variant_name, image, boxes):
    image_dir = output_root / "images" / variant_name
    label_dir = output_root / "labels" / variant_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    output_image = image_dir / f"{image_path.stem}__{variant_name}{image_path.suffix.lower()}"
    output_label = label_dir / f"{image_path.stem}__{variant_name}.txt"
    if output_image.exists() or output_label.exists():
        raise FileExistsError(f"Robust test output already exists: {output_image}")
    success, encoded = cv2.imencode(output_image.suffix, image)
    if not success:
        raise ValueError(f"Could not encode output image: {output_image}")
    output_image.write_bytes(encoded.tobytes())
    output_label.write_text(
        "".join(f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}\n"
                for class_id, xc, yc, width, height in boxes),
        encoding="utf-8",
    )


def _angle_name(angle):
    return f"rotate_{'p' if angle > 0 else 'm'}{abs(angle):g}".replace(".", "p")


def _generate(args):
    _, image_paths = load_split(args.data, "test")
    if args.tilt_fraction <= 0 or args.tilt_fraction >= 0.5:
        raise ValueError("--tilt-fraction must be greater than 0 and less than 0.5")
    if not args.rotate_angles or any(angle == 0 for angle in args.rotate_angles):
        raise ValueError("--rotate-angles must contain non-zero angles")
    if len(set(args.rotate_angles)) != len(args.rotate_angles):
        raise ValueError("--rotate-angles must not contain duplicates")
    if not args.blur_kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in args.blur_kernels):
        raise ValueError("--blur-kernels must contain positive odd integers")
    if len(set(args.blur_kernels)) != len(args.blur_kernels):
        raise ValueError("--blur-kernels must not contain duplicates")

    output_root = args.out.resolve()
    generated = 0
    for image_path in image_paths:
        image = _read_image(image_path)
        height, width = image.shape[:2]
        annotation_path = label_path(image_path)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing test labels: {annotation_path}")
        boxes = read_labels(image_path)
        variants = []

        for angle in args.rotate_angles:
            matrix = _rotation_matrix(width, height, angle)
            warped = cv2.warpPerspective(
                image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101
            )
            variants.append((_angle_name(angle), warped, _transform_boxes(boxes, matrix, width, height)))

        for far_edge in ("top", "bottom"):
            matrix = _tilt_matrix(width, height, args.tilt_fraction, far_edge)
            warped = cv2.warpPerspective(
                image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101
            )
            variants.append((
                f"tilt_{far_edge}",
                warped,
                _transform_boxes(boxes, matrix, width, height),
            ))

        for kernel in args.blur_kernels:
            variants.append((
                f"blur_k{kernel}",
                cv2.GaussianBlur(image, (kernel, kernel), 0),
                boxes,
            ))

        for name, transformed_image, transformed_boxes in variants:
            _write_variant(output_root, image_path, name, transformed_image, transformed_boxes)
            generated += 1

    print(f"Generated {generated} robust test images from {len(image_paths)} test images: {output_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO / "configs" / "data_board.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "dataset" / "pcb_robust_testset")
    parser.add_argument("--rotate-angles", type=float, nargs="+", default=[-10, 10])
    parser.add_argument("--tilt-fraction", type=float, default=0.08)
    parser.add_argument("--blur-kernels", type=int, nargs="+", default=[5, 9])
    _generate(parser.parse_args())


if __name__ == "__main__":
    main()
