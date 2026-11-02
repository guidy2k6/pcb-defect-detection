
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


def align_and_xor(
    baseline_path: Path,
    defect_path: Path,
    lowe_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Căn chỉnh ảnh bằng SIFT (thay cho SURF của [1]) và tính hiệu ảnh (XOR)."""
    # 1. Đọc ảnh xám
    img_base_gray = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE)
    img_def_gray = cv2.imread(str(defect_path), cv2.IMREAD_GRAYSCALE)

    if img_base_gray is None or img_def_gray is None:
        raise FileNotFoundError(
            f"Không thể đọc ảnh: {baseline_path} hoặc {defect_path}"
        )

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
    diff_xor = cv2.bitwise_xor(img_base_gray, aligned_def_gray)

    # Tạo ảnh overlay màu quan sát trực quan
    img_base_color = cv2.imread(str(baseline_path), cv2.IMREAD_COLOR)
    img_def_color = cv2.imread(str(defect_path), cv2.IMREAD_COLOR)
    if img_base_color is None or img_def_color is None:
        raise FileNotFoundError(
            f"Không thể đọc ảnh màu: {baseline_path} hoặc {defect_path}"
        )
    aligned_def_color = cv2.warpPerspective(
        img_def_color, H, (w, h)
    )
    overlay = cv2.addWeighted(img_base_color, 0.5, aligned_def_color, 0.5, 0)

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

    selected_pairs = pairs[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for baseline_path, defect_path in selected_pairs:
        _, _, _, overlay = align_and_xor(baseline_path, defect_path)
        output_path = output_dir / (
            f"{baseline_path.stem}__{defect_path.stem}_overlay.jpg"
        )
        if not cv2.imwrite(str(output_path), overlay):
            raise OSError(f"Không thể lưu ảnh overlay: {output_path}")
        saved_paths.append(output_path)
        print(f"Saved: {output_path}")

    print(f"Done: saved {len(saved_paths)}/{limit} overlays.")
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
