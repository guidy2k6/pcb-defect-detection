"""Evaluate baseline XOR boxes on the board-level test split."""

import argparse
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from analyze_errors import match_image
from baseline_xor import IMAGE_EXTENSIONS, process_pair


REPO = Path(__file__).resolve().parents[1]
EVALUATION_NOTE = (
    "Đánh giá có/không phát hiện lỗi; không đánh giá phân loại lớp lỗi. "
    "Hộp ghép 1-1 khi IoU >= ngưỡng; "
    "count_error_pct lưu Pd theo Eq. (9): Pd = |d - a| / a * 100%, "
    "d = tổng số vùng lỗi phát hiện, a = tổng số vùng lỗi thực tế; "
    "để trống nếu a = 0. Nguồn: Wei et al., A PCB Dataset for Defects "
    "Detection and Classification, arXiv:1901.08204v1, Eq. (9), "
    "https://arxiv.org/abs/1901.08204. "
    "Cặp căn chỉnh thất bại được tính là bỏ sót toàn bộ nhãn."
)


def _count_error_pct(num_pred: int, num_gt: int) -> float | None:
    if num_gt == 0:
        return None
    return abs(num_pred - num_gt) / num_gt * 100


def _read_test_boards(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boards = manifest.get("test")
    if not isinstance(boards, list) or not boards:
        raise ValueError(f"Manifest không có danh sách test hợp lệ: {manifest_path}")
    return {str(board) for board in boards}


def _annotation_index(annotation_root: Path) -> dict[str, Path]:
    if not annotation_root.is_dir():
        raise NotADirectoryError(
            f"Không tìm thấy thư mục annotation: {annotation_root}"
        )
    annotations: dict[str, Path] = {}
    for path in sorted(annotation_root.rglob("*.xml")):
        key = path.stem.casefold()
        if key in annotations:
            raise ValueError(
                f"Có nhiều XML cùng tên {path.stem!r}: "
                f"{annotations[key]} và {path}"
            )
        annotations[key] = path
    return annotations


def _read_ground_truth(
    annotation_path: Path,
) -> tuple[np.ndarray, tuple[int, int]]:
    root = ET.parse(annotation_path).getroot()
    width_text = root.findtext("size/width")
    height_text = root.findtext("size/height")
    if not width_text or not height_text:
        raise ValueError(f"Thiếu kích thước ảnh trong annotation: {annotation_path}")
    image_width, image_height = int(width_text), int(height_text)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Kích thước ảnh không hợp lệ: {annotation_path}")

    boxes = []
    for obj in root.findall("object"):
        bounds = obj.find("bndbox")
        if bounds is None:
            raise ValueError(f"Object thiếu bndbox trong {annotation_path}")
        values = [
            bounds.findtext(key)
            for key in ("xmin", "ymin", "xmax", "ymax")
        ]
        if any(value is None for value in values):
            raise ValueError(f"Bndbox thiếu tọa độ trong {annotation_path}")
        x1, y1, x2, y2 = map(float, values)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Bndbox không hợp lệ trong {annotation_path}")
        boxes.append((x1, y1, x2, y2))

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), (
        image_width,
        image_height,
    )


def _align_ground_truth(
    boxes: np.ndarray,
    annotation_size: tuple[int, int],
    image_shape: tuple[int, ...],
    homography: np.ndarray,
) -> np.ndarray:
    if not len(boxes):
        return np.empty((0, 4), dtype=np.float32)

    image_height, image_width = image_shape[:2]
    annotation_width, annotation_height = annotation_size
    scaled = boxes.copy()
    scaled[:, (0, 2)] *= image_width / annotation_width
    scaled[:, (1, 3)] *= image_height / annotation_height

    corners = np.stack(
        (
            scaled[:, [0, 1]],
            scaled[:, [2, 1]],
            scaled[:, [2, 3]],
            scaled[:, [0, 3]],
        ),
        axis=1,
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 4, 2)
    aligned = np.column_stack(
        (
            transformed[:, :, 0].min(axis=1),
            transformed[:, :, 1].min(axis=1),
            transformed[:, :, 0].max(axis=1),
            transformed[:, :, 1].max(axis=1),
        )
    )
    aligned[:, (0, 2)] = np.clip(aligned[:, (0, 2)], 0, image_width)
    aligned[:, (1, 3)] = np.clip(aligned[:, (1, 3)], 0, image_height)
    return aligned.astype(np.float32)


def _prediction_xyxy(
    boxes: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    return np.asarray(
        [(x, y, x + width, y + height) for x, y, width, height, _ in boxes],
        dtype=np.float32,
    ).reshape(-1, 4)


def evaluate_test_split(
    dataset_root: Path,
    manifest_path: Path,
    output_path: Path,
    iou_threshold: float = 0.5,
) -> list[dict[str, object]]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold phải nằm trong khoảng [0, 1]")

    test_boards = _read_test_boards(manifest_path)
    baseline_dir = dataset_root / "PCB_USED"
    image_root = dataset_root / "images"
    annotation_root = dataset_root / "Annotations"
    if not baseline_dir.is_dir():
        raise NotADirectoryError(f"Không tìm thấy ảnh baseline: {baseline_dir}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Không tìm thấy thư mục ảnh lỗi: {image_root}")

    baselines = {
        path.stem: path
        for path in baseline_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    defect_images = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.split("_", 1)[0] in test_boards
    )
    if not defect_images:
        raise ValueError(f"Không có ảnh test cho các bo: {sorted(test_boards)}")

    annotations = _annotation_index(annotation_root)
    rows: list[dict[str, object]] = []
    for defect_path in defect_images:
        board_id = defect_path.stem.split("_", 1)[0]
        baseline_path = baselines.get(board_id)
        if baseline_path is None:
            raise FileNotFoundError(
                f"Không có ảnh baseline cho board {board_id}: {defect_path}"
            )
        annotation_path = annotations.get(defect_path.stem.casefold())
        if annotation_path is None:
            raise FileNotFoundError(
                f"Không tìm thấy XML nhãn cho ảnh: {defect_path}"
            )

        gt_boxes, annotation_size = _read_ground_truth(annotation_path)
        started = time.perf_counter()
        try:
            images, detected_boxes, homography = process_pair(
                baseline_path, defect_path
            )
            gt_xyxy = _align_ground_truth(
                gt_boxes,
                annotation_size,
                images["defect_gray"].shape,
                homography,
            )
            pred_xyxy = _prediction_xyxy(detected_boxes)
            # The baseline has no class scores; stable equal scores preserve box order.
            pairs, missed, extra = match_image(
                gt_xyxy,
                np.zeros(len(gt_xyxy), dtype=int),
                pred_xyxy,
                np.zeros(len(pred_xyxy), dtype=int),
                np.ones(len(pred_xyxy), dtype=float),
                iou_threshold,
            )
            status = "ok"
            error = ""
            true_positive = len(pairs)
            false_positive = len(extra)
            false_negative = len(missed)
            detected_count = len(pred_xyxy)
        except (ValueError, FileNotFoundError, cv2.error) as err:
            status = "alignment_failed"
            error = str(err)
            true_positive = 0
            false_positive = 0
            false_negative = len(gt_boxes)
            detected_count = 0

        elapsed = time.perf_counter() - started
        count_error_pct = _count_error_pct(detected_count, len(gt_boxes))
        rows.append({
            "record_type": "image",
            "image": defect_path.name,
            "board_id": board_id,
            "annotation": str(annotation_path),
            "iou_threshold": iou_threshold,
            "ground_truth_count": len(gt_boxes),
            "detected_count": detected_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "count_error_pct": count_error_pct,
            "processing_time_seconds": f"{elapsed:.6f}",
            "status": status,
            "error": error,
            "evaluation_note": EVALUATION_NOTE,
        })
        print(
            f"{defect_path.name}: TP={true_positive}, FP={false_positive}, "
            f"FN={false_negative}, {elapsed:.3f}s ({status})"
        )

    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "ground_truth_count",
            "detected_count",
            "true_positive",
            "false_positive",
            "false_negative",
        )
    }
    total_time = sum(float(row["processing_time_seconds"]) for row in rows)
    total_count_error_pct = _count_error_pct(
        totals["detected_count"], totals["ground_truth_count"]
    )
    rows.append({
        "record_type": "summary",
        "image": "ALL",
        "board_id": ",".join(sorted(test_boards)),
        "annotation": "",
        "iou_threshold": iou_threshold,
        **totals,
        "count_error_pct": total_count_error_pct,
        "processing_time_seconds": f"{total_time:.6f}",
        "status": (
            "complete"
            if all(row["status"] == "ok" for row in rows)
            else "completed_with_alignment_failures"
        ),
        "error": "",
        "evaluation_note": EVALUATION_NOTE,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_type",
        "image",
        "board_id",
        "annotation",
        "iou_threshold",
        "ground_truth_count",
        "detected_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "count_error_pct",
        "processing_time_seconds",
        "status",
        "error",
        "evaluation_note",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = rows[-1]
    print(
        f"Overall: TP={summary['true_positive']}, "
        f"FP={summary['false_positive']}, "
        f"FN={summary['false_negative']}, "
        f"count error={summary['count_error_pct']}% "
        f"across {len(defect_images)} images."
    )
    print(f"Saved evaluation: {output_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("PCB_DATASET/PCB_DATASET"),
        help="Dataset root containing PCB_USED, images, and Annotations",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/board_split.json"),
        help="Board-level split manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline_test.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for one-to-one matching (default: 0.5)",
    )
    args = parser.parse_args()
    evaluate_test_split(
        args.dataset_root,
        args.manifest,
        args.output,
        args.iou,
    )


if __name__ == "__main__":
    main()
