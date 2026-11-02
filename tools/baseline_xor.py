
import argparse
from collections.abc import Iterable
from pathlib import Path
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
    return cv2.bitwise_xor(binary_base, binary_def)


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


def align_and_xor(
    baseline_path: Path,
    defect_path: Path,
    lowe_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Căn chỉnh ảnh bằng SIFT (thay cho SURF của [1]) và tính hiệu ảnh (XOR)."""
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

    # 6. Phép toán XOR / Hiệu ảnh để tìm vùng lỗi
    diff_xor = binarize_and_xor(img_base_gray, aligned_def_gray)
    diff_xor, bboxes = post_process_and_extract_bboxes(diff_xor)

    # Tạo ảnh overlay màu quan sát trực quan
    aligned_def_color = cv2.warpPerspective(
        img_def_color, H, (w, h)
    )
    overlay = cv2.addWeighted(img_base_color, 0.5, aligned_def_color, 0.5, 0)
    for x, y, box_width, box_height, _ in bboxes:
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width - 1, y + box_height - 1),
            (0, 0, 255),
            2,
        )

    return img_base_gray, aligned_def_gray, diff_xor, overlay


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Không tìm thấy thư mục ảnh: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_overlay_pairs(
    baseline_dir: Path,
    defect_dir: Path,
    output_dir: Path,
    limit: int = 5,
) -> list[Path]:
    """Căn chỉnh tối đa limit cặp ảnh và lưu ảnh overlay."""
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
    saved_paths = []
    for baseline_path, defect_path in selected_pairs:
        try:
            _, _, _, overlay = align_and_xor(baseline_path, defect_path)
        except ValueError as err:
            print(
                f"Failed: {baseline_path.name} + {defect_path.name}: {err}"
            )
            continue

        output_path = output_dir / (
            f"{baseline_path.stem}__{defect_path.stem}_overlay.jpg"
        )
        if not cv2.imwrite(str(output_path), overlay):
            raise OSError(f"Không thể lưu ảnh overlay: {output_path}")
        saved_paths.append(output_path)
        success_count += 1
        print(f"Success: {baseline_path.name} + {defect_path.name}")

    success_rate = success_count / total_run * 100
    print(
        f"Alignment success rate: {success_count}/{total_run} "
        f"- {success_rate:.2f}%."
    )
    print(f"Saved {len(saved_paths)}/{total_run} overlays.")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Căn chỉnh ảnh PCB bằng SIFT và lưu ảnh overlay."
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
        default=Path("output/overlays"),
        help="Thư mục lưu overlay (mặc định: output/overlays)",
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
    )


if __name__ == "__main__":
    main()
