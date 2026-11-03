import argparse
import csv
from pathlib import Path

import cv2

from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[1]

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_images(data_dir: str | Path) -> list[Path]:
    """
    Lấy danh sách ảnh trong thư mục đầu vào,
    bao gồm cả các thư mục con.
    """
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {data_dir}"
        )

    images = []

    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    if not images:
        raise ValueError(
            f"Không tìm thấy ảnh trong thư mục: {data_dir}"
        )

    return images


def run_inference(
    data_dir: str | Path,
    save_dir: str | Path,
    weights: str | Path,
    conf: float,
) -> None:
    """
    Chạy mô hình YOLO để phát hiện lỗi trên các ảnh.
    """

    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    weights = Path(weights)

    if not weights.is_file():
        raise FileNotFoundError(f"Không tìm thấy file weights: {weights}")

    # Tạo thư mục lưu kết quả
    image_save_dir = save_dir / "images"
    image_save_dir.mkdir(parents=True, exist_ok=True)

    # Nạp mô hình YOLOv8s
    print(f"Đang nạp mô hình: {weights}")
    model = YOLO(str(weights))

    # Lấy danh sách ảnh
    image_paths = get_images(data_dir)

    print(f"Tìm thấy {len(image_paths)} ảnh.")
    print(f"Confidence threshold: {conf}")

    # Các kết quả sẽ được lưu vào danh sách này
    all_predictions = []

    for index, image_path in enumerate(image_paths, start=1):

        print(
            f"[{index}/{len(image_paths)}] "
            f"Đang xử lý: {image_path.name}"
        )

        # Đọc ảnh
        image = cv2.imread(str(image_path))

        if image is None:
            raise OSError(f"Không thể đọc ảnh: {image_path}")

        # Chạy dự đoán
        results = model.predict(
            source=image,
            conf=conf,
            verbose=False
        )

        result = results[0]

        # Lấy ảnh đã vẽ bounding box
        annotated_image = result.plot()

        # Giữ nguyên cấu trúc thư mục con
        relative_path = image_path.relative_to(data_dir)

        output_path = image_save_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Lưu ảnh kết quả
        if not cv2.imwrite(str(output_path), annotated_image):
            raise OSError(f"Không thể lưu ảnh kết quả: {output_path}")

        # Lấy thông tin các bounding box
        boxes = result.boxes

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for box_index in range(len(xyxy)):

                x1, y1, x2, y2 = xyxy[box_index]

                class_id = class_ids[box_index]
                confidence = confidences[box_index]

                class_name = model.names[class_id]

                all_predictions.append({
                    "image": str(relative_path),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                })

    # Xuất kết quả CSV
    csv_path = save_dir / "predictions.csv"

    fieldnames = [
        "image",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_predictions)

    print("\nHoàn thành inference!")
    print(f"Ảnh kết quả: {image_save_dir}")
    print(f"File CSV: {csv_path}")
    print(f"Tổng số bounding box: {len(all_predictions)}")


def main():

    parser = argparse.ArgumentParser(
        description="YOLOv8 PCB defect inference"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Thư mục chứa ảnh đầu vào"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Thư mục lưu kết quả"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=str(REPO / "best.pt"),
        help="Đường dẫn đến file trọng số"
    )

    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf phải nằm trong khoảng từ 0 đến 1")

    run_inference(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        weights=args.weights,
        conf=args.conf
    )


if __name__ == "__main__":
    main()