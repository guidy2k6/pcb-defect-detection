# Tham so cac lan huan luyen

| Ngay gio | Run | Model | Data | imgsz | batch | epochs | patience | seed | ultralytics |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-31 09:13 | pcb_v8n_random | yolov8n.pt | C:/Users/Admin/Downloads/pcb_yolo/data.yaml | 1024 | 8 | 100 | 30 | 0 | 8.4.194 |


## Ghi chú

- `pcb_v8s` (chạy 06/10/2026, ngoài repo): `yolov8s.pt`, chia ngẫu nhiên, imgsz 1024, batch 8, epochs 100, patience 30, seed 0. Kết quả: mAP50 0,967, mAP50-95 0,539. Phiên bản Ultralytics và Python của lần chạy này không ghi lại, nên có thể khác môi trường hiện tại.
- Môi trường các run trong repo: Ultralytics 8.4.194, Python 3.14.6, torch 2.14.1+cu126, RTX 3050 6 GB.
- `pcb_v8n_random` chạy với biến môi trường `KMP_DUPLICATE_LIB_OK=TRUE` để tránh lỗi OMP #15 (trùng `libiomp5md.dll` giữa PyTorch và conda). Lần chạy đầu bị lỗi này và dừng ngay sau bước vẽ `labels.jpg`, nên đã chạy lại từ đầu và bỏ dòng log của lần lỗi.
- `pcb_v8n_random`: dừng sớm ở epoch 98 (best epoch 68, patience 30). Kết quả trên val chia ngẫu nhiên: P 0,952, R 0,945, mAP50 0,964, mAP50-95 0,528. Thời gian huấn luyện 3,5 giờ.

- Môi trường: anaconda base, Python 3.14.6, Ultralytics 8.4.194, torch 2.14.1+cu126.
- Số liệu dùng cho slide được đo lại bằng `tools/eval.py` (batch 8, imgsz 1024, val chia ngẫu nhiên): v8n P 0,970, R 0,933, mAP50 0,967, mAP50-95 0,529, 13,6 ms/ảnh; v8s P 0,973, R 0,947, mAP50 0,967, mAP50-95 0,544, 24,9 ms/ảnh. Lệch nhẹ so với số in cuối lúc train.
- `analyze_errors.py`: conf 0,25; IoU ghép 0,5; imgsz 1024.