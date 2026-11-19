"""Generate a YOLO split with random rotation and blur, retaining some clean images.

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


def _transform_boxes(boxes, matrix, width, height, output_width, output_height):
    transformed = []
    for class_id, xc, yc, box_width, box_height in boxes:
        x1, x2 = (xc - box_width / 2) * width, (xc + box_width / 2) * width
        y1, y2 = (yc - box_height / 2) * height, (yc + box_height / 2) * height
        corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
            dtype=np.float32,
        )
        points = cv2.perspectiveTransform(corners - 0.5, matrix)[0] + 0.5
        x_min = float(np.clip(points[:, 0].min(), 0, output_width))
        x_max = float(np.clip(points[:, 0].max(), 0, output_width))
        y_min = float(np.clip(points[:, 1].min(), 0, output_height))
        y_max = float(np.clip(points[:, 1].max(), 0, output_height))
        if x_max > x_min and y_max > y_min:
            transformed.append((
                class_id,
                (x_min + x_max) / (2 * output_width),
                (y_min + y_max) / (2 * output_height),
                (x_max - x_min) / output_width,
                (y_max - y_min) / output_height,
            ))
    return transformed


def _rotation_matrix(width, height, angle):
    affine = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    cosine, sine = abs(affine[0, 0]), abs(affine[0, 1])
    output_width = int(np.ceil((width - 1) * cosine + (height - 1) * sine - 1e-9)) + 1
    output_height = int(np.ceil((height - 1) * cosine + (width - 1) * sine - 1e-9)) + 1
    affine[0, 2] += (output_width - 1) / 2 - (width - 1) / 2
    affine[1, 2] += (output_height - 1) / 2 - (height - 1) / 2
    return np.vstack((affine, (0, 0, 1))).astype(np.float32), output_width, output_height


def _read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _write_image_and_label(output_root, split, image_path, image, boxes):
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    output_image = image_dir / f"{image_path.stem}{image_path.suffix.lower()}"
    output_label = label_dir / f"{image_path.stem}.txt"
    if output_image.exists() or output_label.exists():
        raise FileExistsError(f"Output already exists: {output_image}")
    success, encoded = cv2.imencode(output_image.suffix, image)
    if not success:
        raise ValueError(f"Could not encode output image: {output_image}")
    output_image.write_bytes(encoded.tobytes())
    output_label.write_text(
        "".join(
            f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}\n"
            for class_id, xc, yc, width, height in boxes
        ),
        encoding="utf-8",
    )


def _generate_split(args, split, rng):
    _, image_paths = load_split(args.data, split)
    output_root = args.out.resolve()
    clean_total = min(
        len(image_paths),
        max(1, round(len(image_paths) * args.clean_ratio)) if args.clean_ratio > 0 else 0,
    )
    clean_indices = set(rng.choice(len(image_paths), size=clean_total, replace=False))
    clean_count = 0
    for index, image_path in enumerate(image_paths):
        image = _read_image(image_path)
        height, width = image.shape[:2]
        annotation_path = label_path(image_path)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing labels: {annotation_path}")
        boxes = read_labels(image_path)

        if index in clean_indices:
            output_image, output_boxes = image, boxes
            clean_count += 1
        else:
            angle = float(rng.uniform(args.angle_min, args.angle_max))
            angle *= -1 if rng.random() < 0.5 else 1
            matrix, output_width, output_height = _rotation_matrix(width, height, angle)
            output_image = cv2.warpPerspective(
                image, matrix, (output_width, output_height), borderMode=cv2.BORDER_CONSTANT
            )
            output_boxes = _transform_boxes(
                boxes, matrix, width, height, output_width, output_height
            )
            sigma = float(rng.uniform(args.blur_sigma_min, args.blur_sigma_max))
            output_image = cv2.GaussianBlur(
                output_image, (0, 0), sigmaX=sigma, sigmaY=sigma
            )

        _write_image_and_label(output_root, split, image_path, output_image, output_boxes)

    print(
        f"Generated {len(image_paths)} images ({clean_count} unchanged) "
        f"from {split}: {output_root}"
    )


def _generate(args):
    if not 0 <= args.clean_ratio <= 1:
        raise ValueError("--clean-ratio must be between 0 and 1")
    if args.angle_min <= 0 or args.angle_max < args.angle_min or args.angle_max >= 90:
        raise ValueError("--angle-min and --angle-max must satisfy 0 < min <= max < 90")
    if args.blur_sigma_min <= 0 or args.blur_sigma_max < args.blur_sigma_min:
        raise ValueError("--blur-sigma-min and --blur-sigma-max must satisfy 0 < min <= max")

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    rng = np.random.default_rng(args.seed)
    for split in splits:
        _generate_split(args, split, rng)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=REPO / "configs" / "data_board.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "dataset" / "pcb_robust")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--clean-ratio", type=float, default=0.2)
    parser.add_argument("--angle-min", type=float, default=10)
    parser.add_argument("--angle-max", type=float, default=20)
    parser.add_argument("--blur-sigma-min", type=float, default=1)
    parser.add_argument("--blur-sigma-max", type=float, default=4)
    parser.add_argument("--seed", type=int, default=42)
    _generate(parser.parse_args())


if __name__ == "__main__":
    main()
