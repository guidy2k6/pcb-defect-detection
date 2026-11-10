# Tham so cac lan huan luyen

## Ghi chú

- `pcb_v8s` (chạy 06/10/2026, ngoài repo): `yolov8s.pt`, chia ngẫu nhiên, imgsz 1024, batch 8, epochs 100, patience 30, seed 0. Kết quả: mAP50 0,967, mAP50-95 0,539. Phiên bản Ultralytics và Python của lần chạy này không ghi lại, nên có thể khác môi trường hiện tại.
- Môi trường các run trong repo: Ultralytics 8.4.194, Python 3.14.6, torch 2.14.1+cu126, RTX 3050 6 GB.
- `pcb_v8n_random` chạy với biến môi trường `KMP_DUPLICATE_LIB_OK=TRUE` để tránh lỗi OMP #15 (trùng `libiomp5md.dll` giữa PyTorch và conda). Lần chạy đầu bị lỗi này và dừng ngay sau bước vẽ `labels.jpg`, nên đã chạy lại từ đầu và bỏ dòng log của lần lỗi.
- `pcb_v8n_random`: dừng sớm ở epoch 98 (best epoch 68, patience 30). Kết quả trên val chia ngẫu nhiên: P 0,952, R 0,945, mAP50 0,964, mAP50-95 0,528. Thời gian huấn luyện 3,5 giờ.
- Môi trường: anaconda base, Python 3.14.6, Ultralytics 8.4.194, torch 2.14.1+cu126.
- Số liệu dùng cho slide được đo lại bằng `tools/eval.py` (batch 8, imgsz 1024, val chia ngẫu nhiên): v8n P 0,970, R 0,933, mAP50 0,967, mAP50-95 0,529, 13,6 ms/ảnh; v8s P 0,973, R 0,947, mAP50 0,967, mAP50-95 0,544, 24,9 ms/ảnh. Lệch nhẹ so với số in cuối lúc train.
- `analyze_errors.py`: conf 0,25; IoU ghép 0,5; imgsz 1024.
- Test (chạy một lần, `pcb_v8s_board`, bo 08 và 12, `tools/eval.py`): P 0,854, R 0,934, mAP50 0,972, mAP50-95 0,516, 25,7 ms/ảnh (chạy lại bằng --force để sửa tên lớp, số liệu tổng không đổi). Theo tên lớp đã sửa (03/11): precision thấp ở Mouse Bite (0,627) và Short (0,613); recall thấp nhất ở Open Circuit (0,714).
- `pcb_cls_board` (bộ phân loại): `yolov8n-cls.pt`, ảnh cắt 128x128 theo nhãn (lề 50%, `data/io/collect_crops.py`, chia theo bo: train 1749, val 603, test 601 ảnh), batch 64, epochs 50, patience 15. Dừng sớm ở epoch 31, best epoch 16, 0,11 giờ. Top-1: val 1,000, test 0,990 (595 đúng, 6 sai: 5 ảnh Spur nhận thành Spurious Copper, 1 ảnh Mouse Bite nhận thành Open Circuit). Đây là cận trên vì ảnh cắt đúng theo nhãn thật.
- `pcb_v8s_board`: chia theo bo (train 453, val 120, test 120 ảnh; val là bo 07 và 11). Bị ngắt giữa chừng, train tiếp bằng `resume=True` từ epoch 36; thư mục run gốc là `pcb_v8s_board-3`, sau đó đổi tên. Dừng sớm ở epoch 60, best epoch 30 (patience 30). Đo bằng `tools/eval.py` trên val: P 0,975, R 0,958, mAP50 0,978, mAP50-95 0,551, 24,2 ms/ảnh. Phân tích lỗi (`analyze_errors.py`): 603 nhãn, đúng 589, nhầm lớp 3, bỏ sót 11, báo thừa 20.
- `pcb_v8n_board`: máy tắt giữa chừng sau epoch đầu, train tiếp bằng `resume=True` từ `last.pt` (lần resume không ghi thêm dòng vào bảng bên dưới; dòng 12:33 là lần khởi chạy đầu). Dừng sớm ở epoch 78 (best epoch 48, patience 30), 1,9 giờ. Đo bằng `tools/eval.py` trên val theo bo: P 0,973, R 0,917, mAP50 0,972, mAP50-95 0,519, 13,3 ms/ảnh. Phân tích lỗi: 603 nhãn, đúng 566, nhầm lớp 11, bỏ sót 26, báo thừa 28.
- `analyze_errors.py` của `pcb_v8s_board` chạy với `--device cpu` vì predict trên GPU báo OOM lúc đó (eval và train trên GPU không bị lỗi này); `pcb_v8n_board` chạy trên GPU không lỗi.
- Chọn mô hình: `pcb_v8s_board`, dựa trên val theo bo. mAP50-95 0,551 so với 0,519 của v8n_board, bỏ sót 11 so với 26, nhầm lớp 3 so với 11; đổi lại chậm hơn (24,2 so với 13,3 ms/ảnh). Val chỉ có 2 bo và một seed nên chênh lệch chỉ là gợi ý; số kết luận lấy từ test (chạy một lần, xem `results/test_metrics.csv`).

- Sửa tên lớp (03/11/2026): thứ tự lớp trong file nhãn YOLO là abc (0 Missing Hole ... 5 Spurious Copper), còn `data_board.yaml` trước đó ghi thứ tự khác nên tên lớp trong weights và trong `dataset/cls` bị hoán vị. Số liệu tổng không đổi. Đã sửa tên bằng `tools/fix_class_names.py` và đánh giá lại. Kết quả theo lớp trước thời điểm này có tên lớp sai.
- `tools/eval_pipeline.py` (đường XOR): tham số chọn trên val (close 27, min_area 250, grow 2,0; F1 ở IoU 0,5 là 0,618). Test chạy một lần: F1 0,638, phân loại trên box tìm đúng 0,917 (342/373).

## Bảng các lần chạy

| Ngay gio | Run | Model | Data | imgsz | batch | epochs | patience | seed | ultralytics |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-31 09:13 | pcb_v8n_random | yolov8n.pt | C:/Users/Admin/Downloads/pcb_yolo/data.yaml | 1024 | 8 | 100 | 30 | 0 | 8.4.194 |
| 2026-11-02 08:31 | pcb_v8s_board | yolov8s.pt | configs/data_board.yaml | 1024 | 8 | 100 | 30 | 0 | 8.4.194 |
| 2026-11-02 12:33 | pcb_v8n_board | yolov8n.pt | configs/data_board.yaml | 1024 | 8 | 100 | 30 | 0 | 8.4.194 |
| 2026-11-02 15:34 | pcb_cls_board | yolov8n-cls.pt | dataset/cls | 128 | 64 | 50 | 15 | 0 | 8.4.194 |
| 2026-11-02 15:43 | pcb_best_640 | yolov8s.pt | configs/data_board.yaml | 640 | 8 | 100 | 30 | 0 | 8.4.194 |
