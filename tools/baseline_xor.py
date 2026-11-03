
import argparse
import csv
from collections.abc import Iterable
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _board_id(image: str | Path) -> str:
    """Trích xuất ID của bo mạch từ tên tệp."""
    stem = Path(image).stem
    return stem.split("_", 1)[0]


def pair_images_by_board_id(
    baseline_images: Iterable[str | Path],
    defect_images: Iterable[str | Path],
) -> list[tuple[Path, Path]]:
    """Ghép cặp từng ảnh lỗi với ảnh baseline tương ứng theo board ID."""
    baselines: dict[str, Path] = {}
    for image in baseline_images:
        path = Path(image)
        board_id = _board_id(path)
        if board_id in baselines:
            raise ValueError(
                f"Phát hiện nhiều hơn 1 ảnh baseline cho board {board_id!r}"
            )
        baselines[board_id] = path

    pairs = []
    missing_board_ids = set()
    for image in defect_images:
        path = Path(image)
        board_id = _board_id(path)
        baseline = baselines.get(board_id)
        if baseline is None:
            missing_board_ids.add(board_id)
            continue
        pairs.append((baseline, path))

    if missing_board_ids:
        missing = ", ".join(sorted(missing_board_ids))
        raise ValueError(f"Không tìm thấy ảnh baseline cho board ID: {missing}")

    return pairs


def to_gray(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh màu BGR sang ảnh xám."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize_and_xor(
    img_base_gray: np.ndarray,
    img_def_gray: np.ndarray,
    block_size: int = 11,
    c: float = 2,
) -> np.ndarray:
    """Nhị phân hóa thích ứng hai ảnh xám rồi XOR để tìm vùng khác biệt."""
    binary_base, binary_def = binarize_pair(
        img_base_gray, img_def_gray, block_size, c
    )
    return cv2.bitwise_xor(binary_base, binary_def)


def binarize_pair(
    img_base_gray: np.ndarray,
    img_def_gray: np.ndarray,
    block_size: int = 11,
    c: float = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Nhị phân hóa thích ứng hai ảnh xám."""
    if img_base_gray.ndim != 2 or img_def_gray.ndim != 2:
        raise ValueError("Hai đầu vào phải là ảnh xám một kênh")
    if img_base_gray.dtype != np.uint8 or img_def_gray.dtype != np.uint8:
        raise ValueError("Hai ảnh xám phải có kiểu dữ liệu uint8")
    if img_base_gray.shape != img_def_gray.shape:
        raise ValueError("Hai ảnh xám phải có cùng kích thước")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size phải là số lẻ lớn hơn hoặc bằng 3")

    binary_base = cv2.adaptiveThreshold(
        img_base_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    binary_def = cv2.adaptiveThreshold(
        img_def_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    return binary_base, binary_def


def post_process_and_extract_bboxes(
    diff_bin: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """Khử nhiễu mask nhị phân, bỏ vùng nhỏ và trả về bbox (x, y, w, h, area)."""
    if diff_bin.ndim != 2 or diff_bin.dtype != np.uint8:
        raise ValueError("diff_bin phải là ảnh nhị phân một kênh kiểu uint8")
    if min(diff_bin.shape) < 3:
        raise ValueError("diff_bin phải có chiều cao và chiều rộng tối thiểu là 3")

    denoised = cv2.medianBlur(diff_bin, 3)
    kernel = np.ones((3, 3), dtype=np.uint8)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        denoised, connectivity=8
    )
    clean_mask = np.zeros_like(denoised)
    bboxes = []
    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        if area < 20:
            continue
        clean_mask[labels == label] = 255
        bboxes.append(
            (int(x), int(y), int(width), int(height), int(area))
        )

    return clean_mask, bboxes


def process_pair(
    baseline_path: Path,
    defect_path: Path,
    lowe_ratio: float = 0.75,
) -> tuple[
    dict[str, np.ndarray],
    list[tuple[int, int, int, int, int]],
    np.ndarray,
]:
    """Căn chỉnh hai ảnh, tạo mask XOR và trả các ảnh trung gian."""
    # 1. Đọc ảnh màu và chuyển sang ảnh xám
    img_base_color = cv2.imread(str(baseline_path), cv2.IMREAD_COLOR)
    img_def_color = cv2.imread(str(defect_path), cv2.IMREAD_COLOR)
    if img_base_color is None or img_def_color is None:
        raise FileNotFoundError(
            f"Không thể đọc ảnh: {baseline_path} hoặc {defect_path}"
        )
    img_base_gray = to_gray(img_base_color)
    img_def_gray = to_gray(img_def_color)

    # 2. Phát hiện đặc trưng SIFT
    sift = cv2.SIFT_create()
    kp_base, des_base = sift.detectAndCompute(img_base_gray, None)
    kp_def, des_def = sift.detectAndCompute(img_def_gray, None)
    if des_base is None or des_def is None:
        raise ValueError(
            f"Không tìm thấy đặc trưng SIFT trong "
            f"{baseline_path.name} hoặc {defect_path.name}"
        )

    # 3. Khớp điểm đặc trưng & lọc tỷ lệ Lowe
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50)
    )
    matches = flann.knnMatch(des_def, des_base, k=2)

    good_matches = [
        m
        for neighbors in matches
        if len(neighbors) == 2
        for m, n in [neighbors]
        if m.distance < lowe_ratio * n.distance
    ]

    if len(good_matches) < 10:
        raise ValueError(
            f"Không đủ điểm khớp giữa {baseline_path.name} và {defect_path.name}"
        )

    # 4. Tìm Homography với RANSAC
    src_pts = np.float32(
        [kp_def[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_base[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError(
            f"Không tìm được homography giữa {baseline_path.name} "
            f"và {defect_path.name}"
        )

    # 5. Căn chỉnh ảnh (Warp Perspective)
    h, w = img_base_gray.shape
    aligned_def_gray = cv2.warpPerspective(img_def_gray, H, (w, h))

    # 6. Nhị phân hóa, XOR và lọc nhiễu
    binary_base, binary_def = binarize_pair(img_base_gray, aligned_def_gray)
    diff_xor_raw = cv2.bitwise_xor(binary_base, binary_def)
    diff_xor_filtered, bboxes = post_process_and_extract_bboxes(diff_xor_raw)

    # Tạo ảnh overlay màu để quan sát căn chỉnh và kết quả cuối
    aligned_def_color = cv2.warpPerspective(
        img_def_color, H, (w, h)
    )
    alignment_overlay = cv2.addWeighted(
        img_base_color, 0.5, aligned_def_color, 0.5, 0
    )
    final_boxes = alignment_overlay.copy()
    for x, y, box_width, box_height, _ in bboxes:
        cv2.rectangle(
            final_boxes,
            (x, y),
            (x + box_width - 1, y + box_height - 1),
            (0, 0, 255),
            2,
        )

    return {
        "baseline_gray": img_base_gray,
        "defect_gray": img_def_gray,
        "aligned_defect_gray": aligned_def_gray,
        "alignment_overlay": alignment_overlay,
        "baseline_binary": binary_base,
        "defect_binary": binary_def,
        "xor_raw": diff_xor_raw,
        "after_filter": diff_xor_filtered,
        "final_boxes": final_boxes,
    }, bboxes, H


def align_and_xor(
    baseline_path: Path,
    defect_path: Path,
    lowe_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Giữ giao diện cũ; căn chỉnh ảnh bằng SIFT và tính XOR."""
    images, _, _ = process_pair(baseline_path, defect_path, lowe_ratio)
    return (
        images["baseline_gray"],
        images["aligned_defect_gray"],
        images["after_filter"],
        images["final_boxes"],
    )


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Không tìm thấy thư mục ảnh: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _annotation_index(annotation_dir: Path) -> dict[str, list[Path]]:
    if not annotation_dir.is_dir():
        raise NotADirectoryError(f"Không tìm thấy thư mục annotation: {annotation_dir}")
    annotations: dict[str, list[Path]] = {}
    for path in annotation_dir.rglob("*.xml"):
        annotations.setdefault(path.stem.casefold(), []).append(path)
    return annotations


def _label_box_count(
    defect_path: Path,
    annotations: dict[str, list[Path]],
) -> tuple[Path, int]:

    matches = annotations.get(defect_path.stem.casefold(), [])

    if not matches:
        raise FileNotFoundError(
            f"Không tìm thấy XML nhãn cho ảnh lỗi: {defect_path.name}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Có nhiều XML cùng tên cho ảnh {defect_path.name}: "
            + ", ".join(str(path) for path in matches)
        )

    annotation_path = matches[0]
    root = ET.parse(annotation_path).getroot()

    annotated_filename = root.findtext("filename")

    if (
        annotated_filename
        and Path(annotated_filename).name.casefold()
        != defect_path.name.casefold()
    ):
        raise ValueError(
            f"Tên ảnh trong {annotation_path} không khớp "
            f"với {defect_path.name}"
        )

    # Chỉ đếm object có bounding box
    objects = root.findall("object")

    label_count = sum(
        1 for obj in objects
        if obj.find("bndbox") is not None
    )

    return annotation_path, label_count


def save_overlay_pairs(
    baseline_dir: Path,
    defect_dir: Path,
    output_dir: Path,
    limit: int = 5,
    annotation_dir: Path | None = None,
) -> list[Path]:
    """Chạy tối đa limit cặp, lưu các bước xử lý và đối chiếu số box nhãn."""
    if limit < 1:
        raise ValueError("Số cặp ảnh cần chạy phải lớn hơn 0")

    baseline_images = _image_files(baseline_dir)
    defect_images = _image_files(defect_dir)
    if not baseline_images:
        raise ValueError(f"Không có ảnh baseline trong {baseline_dir}")
    if not defect_images:
        raise ValueError(f"Không có ảnh lỗi trong {defect_dir}")

    baseline_ids = {_board_id(image) for image in baseline_images}
    matched_defects = [
        image for image in defect_images if _board_id(image) in baseline_ids
    ]
    skipped_count = len(defect_images) - len(matched_defects)
    if skipped_count:
        print(f"Skipped {skipped_count} defect images without a matching baseline.")

    pairs = pair_images_by_board_id(baseline_images, matched_defects)
    if not pairs:
        raise ValueError("Không tìm thấy cặp ảnh baseline/ảnh lỗi phù hợp")

    total_pairs = len(pairs)
    if total_pairs <= limit:
        selected_pairs = pairs
    else:
        sample_indices = np.linspace(
            0, total_pairs - 1, num=limit, dtype=int
        )
        selected_pairs = [pairs[int(index)] for index in sample_indices]

    total_run = len(selected_pairs)
    success_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir = (
        annotation_dir
        if annotation_dir is not None
        else defect_dir.parent.parent / "Annotations"
    )
    annotations = _annotation_index(annotation_dir)
    summary_rows = []
    failure_rows = []
    saved_paths = []

    for baseline_path, defect_path in selected_pairs:

        pair_name = f"{baseline_path.stem}__{defect_path.stem}"
        pair_dir = output_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        # 1. Đọc số box nhãn
        annotation_path = None
        label_count = None
        annotation_error = ""

        try:
            annotation_path, label_count = _label_box_count(
                defect_path, annotations
            )

        except (FileNotFoundError, ValueError, ET.ParseError, OSError) as err:
            annotation_error = str(err)
            print(f"Annotation warning: {pair_name}: {err}")

        # 2. Căn chỉnh và xử lý ảnh
        try:
            images, bboxes, _ = process_pair(
                baseline_path, defect_path
            )

        except (ValueError, FileNotFoundError, cv2.error) as err:

            reason = str(err)

            # Lưu ảnh xám kể cả khi căn chỉnh thất bại
            for image_path, gray_name in (
                (baseline_path, "baseline_gray.png"),
                (defect_path, "defect_gray.png"),
            ):
                color_image = cv2.imread(
                    str(image_path), cv2.IMREAD_COLOR
                )

                if color_image is None:
                    raise FileNotFoundError(
                        f"Không thể đọc ảnh: {image_path}"
                    )

                gray_path = pair_dir / gray_name

                if not cv2.imwrite(
                    str(gray_path), to_gray(color_image)
                ):
                    raise OSError(f"Không thể lưu ảnh: {gray_path}")

            # Ghi nhận cặp căn chỉnh thất bại
            failure_rows.append({
                "baseline_image": baseline_path.name,
                "defect_image": defect_path.name,
                "reason": reason,
            })

            summary_rows.append({
                "baseline_image": baseline_path.name,
                "defect_image": defect_path.name,
                "annotation": str(annotation_path or ""),
                "label_box_count": (
                    label_count if label_count is not None else ""
                ),
                "detected_box_count": "",
                "difference": "",
                "annotation_status": (
                    "found" if label_count is not None else "missing"
                ),
                "annotation_error": annotation_error,
                "status": "alignment_failed",
            })

            print(f"Alignment failed: {pair_name}: {reason}")
            continue

        # 3. Lưu tất cả ảnh trung gian
        for image_name, image in images.items():

            extension = ".jpg" if image.ndim == 3 else ".png"
            output_path = pair_dir / f"{image_name}{extension}"

            if not cv2.imwrite(str(output_path), image):
                raise OSError(f"Không thể lưu ảnh: {output_path}")

            if image_name == "final_boxes":
                saved_paths.append(output_path)

        # 4. Đếm số box thuật toán phát hiện
        detected_count = len(bboxes)

        # Chỉ tính chênh lệch khi có nhãn
        difference = (
            detected_count - label_count
            if label_count is not None
            else ""
        )

        # 5. Ghi kết quả
        summary_rows.append({
            "baseline_image": baseline_path.name,
            "defect_image": defect_path.name,
            "annotation": str(annotation_path or ""),
            "label_box_count": (
                label_count if label_count is not None else ""
            ),
            "detected_box_count": detected_count,
            "difference": difference,
            "annotation_status": (
                "found" if label_count is not None else "missing"
            ),
            "annotation_error": annotation_error,
            "status": "ok" if label_count is not None else "annotation_missing",
        })

        success_count += 1

        print(
            f"{pair_name}: detected {detected_count} boxes; "
            f"labels {label_count}; difference {difference}"
        )


    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "baseline_image",
            "defect_image",
            "annotation",
            "label_box_count",
            "detected_box_count",
            "difference",
            "annotation_status",
            "annotation_error",
            "status",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    failures_path = output_dir / "alignment_failures.csv"
    with failures_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["baseline_image", "defect_image", "reason"]
        )
        writer.writeheader()
        writer.writerows(failure_rows)

    success_rate = success_count / total_run * 100
    print(
        f"Alignment success rate: {success_count}/{total_run} "
        f"- {success_rate:.2f}%."
    )
    print(f"Saved intermediate images under: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Alignment failures: {failures_path} ({len(failure_rows)})")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Căn chỉnh ảnh PCB bằng SIFT, lưu các bước XOR và so số box."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Thư mục ảnh baseline, ví dụ PCB_DATASET/PCB_DATASET/PCB_USED",
    )
    parser.add_argument(
        "--defect-dir",
        type=Path,
        required=True,
        help="Thư mục ảnh lỗi của một loại lỗi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/baseline_xor"),
        help="Thư mục lưu ảnh từng bước và CSV kết quả",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Thư mục XML nhãn (mặc định: thư mục Annotations cạnh images)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Số cặp ảnh tối đa cần chạy (mặc định: 5)",
    )
    args = parser.parse_args()
    save_overlay_pairs(
        args.baseline_dir,
        args.defect_dir,
        args.output_dir,
        args.limit,
        args.annotation_dir,
    )


if __name__ == "__main__":
    main()
