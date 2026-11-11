# Báo cáo tiến độ và kết quả đề tài

**Đề tài:** Phát hiện và phân loại lỗi trên bảng mạch in PCB  
**Ngày cập nhật:** 11/11/2026  
**Trạng thái:** Đã có dữ liệu chia theo bo, kết quả YOLO trên validation và test, phân tích lỗi (gồm 307 dương tính giả trên test), ngưỡng tin cậy theo lớp, pipeline XOR và demo.

## 1. Tổng quan

### 1.1. Dữ liệu và cách đánh giá

Dự án nhận diện sáu loại lỗi: Missing Hole, Mouse Bite, Open Circuit, Short, Spur và Spurious Copper. Dữ liệu được chia theo ID bo thành 453 ảnh train, 120 ảnh validation và 120 ảnh test (1749, 603 và 601 box). Có 10 bo, chia 6/2/2: validation là bo 07 và 11, test là bo 08 và 12. Mỗi bo nằm trọn trong một tập để tránh cùng một bo xuất hiện ở cả huấn luyện và đánh giá.

Các điểm cần trình bày chính xác:

1. **Nguồn dữ liệu:** quy trình đọc bộ PCB gốc có `images/`, `Annotations/` và `PCB_USED/` (ảnh bo mẫu không lỗi).
2. **Lịch sử đánh giá test:** `results/test_metrics.csv` có kết quả YOLOv8s board trên test. Test được chạy lại một lần bằng `--force` sau khi sửa tên lớp, số tổng không đổi. Sau đó test được dùng để phân tích lỗi (`analyze_errors.py`). Mô hình và ngưỡng đều được chọn trên validation, không chọn theo test.
3. **Rò rỉ theo bo:** split theo ID bo thay vì ngẫu nhiên theo ảnh; báo cáo audit nằm ở `results/data_audit.csv`.
4. **Các mốc so sánh:** YOLOv8n và YOLOv8s, chia ngẫu nhiên và theo bo, imgsz 640, và baseline XOR. YOLO và XOR được đo bằng metric khác nhau nên không so trực tiếp mAP với F1.

### 1.2. Sản phẩm và tình trạng

| Hạng mục | Tình trạng |
|---|---|
| Mã nguồn và hướng dẫn chạy | `README.md`, `requirements.txt`, `configs/`, `data/io/`, `tools/`, `demo/`, `tests/` |
| Dataset YOLO đã chuyển đổi | Không lưu trong git; tạo lại từ bộ gốc bằng `data/io/divide_data.py` |
| Weights | Không lưu trong git (`*.pt` bị chặn); lưu trên Hugging Face, demo tải qua `HF_MODEL_REPO` |
| Kết quả và phân tích | CSV, biểu đồ và ảnh trong `results/`; bảng tự sinh ở `reports/bang_so_lieu.md` |
| Demo | `demo/app.py` (Gradio): YOLO một ảnh, và tùy chọn so với ảnh mẫu (SIFT + XOR) |

### 1.3. Hai hướng triển khai

Hướng ảnh mẫu: căn chỉnh ảnh lỗi với ảnh PCB chuẩn, nhị phân hóa, XOR, lọc vùng khác biệt và phân loại vùng lỗi (`tools/baseline_xor.py`, `tools/pipeline.py`, `tools/eval_pipeline.py`). Hướng YOLO: detector một giai đoạn (`tools/train.py`, `tools/eval.py`, `tools/analyze_errors.py`), dùng làm mốc so sánh và đề xuất bổ sung.

### 1.4. Quy trình xử lý và file tương ứng

| Khối công việc | Thực hiện trong repo | Đầu ra chính |
|---|---|---|
| Đọc XML, chuyển VOC sang YOLO, chia theo bo | `data/io/divide_data.py` (gọi `convert_voc_to_yolo.py`, `check_xml.py`) | `dataset/pcb_boards_yolo/`, `configs/data_board.yaml`, `configs/board_split.json`, `results/data_audit.csv` |
| Huấn luyện detector | `tools/train.py` | `runs/detect/<run>/weights/`, ghi thông số vào `configs/train_params.md` |
| Sửa tên lớp trong weights | `tools/fix_class_names.py` | Weights mới, không train lại |
| Đánh giá detector | `tools/eval.py` | `results/<run>_<split>/`, `results/test_metrics.csv` |
| Phân tích lỗi | `tools/analyze_errors.py` | `results/<run>_<split>_errors/` |
| Chọn ngưỡng tin cậy theo lớp (val) | `tools/select_thresholds.py` | `configs/thresholds.json` |
| Cắt vùng dương tính giả để xem | `tools/crop_hotspot_v2.py` | `results/hotspot2_<bo>_<x>_<y>.png` |
| Bảng số liệu tự sinh | `tools/make_report_tables.py` | `reports/bang_so_lieu.md` |
| Tổng hợp các lần eval | `tools/build_summary.py` | `results/summary.csv` |
| Classifier crop | `data/io/collect_crops.py`, `tools/train_cls.py` | `dataset/cls/`, `results/pcb_cls_board*` |
| Đánh giá XOR | `tools/eval_pipeline.py` | `results/pipeline_<split>.csv` |

### 1.5. Nguyên tắc báo cáo kết quả

- Mỗi số liệu truy được về một file dưới `results/` hoặc nhật ký `configs/train_params.md`.
- Chọn mô hình và tham số (kể cả ngưỡng tin cậy) bằng validation; test chỉ dùng báo cáo.
- Tách rõ kết quả crop-classifier trên box nhãn thật khỏi kết quả end-to-end.
- Ghi rõ cách tính: P, R "gộp" là tính trên tất cả box; P, R của Ultralytics là trung bình theo lớp. Hai cách cho số khác nhau, đặc biệt trên test.

## 2. Kho mã và cách chạy

```text
configs/       cấu hình dữ liệu, manifest chia bo, nhật ký train, thresholds.json
data/io/       kiểm tra dữ liệu, chuyển VOC, chia split, tạo crop
tools/         train, eval, phân tích lỗi, ngưỡng, pipeline XOR, tiện ích
demo/          giao diện Gradio, service, ví dụ
tests/         kiểm thử cho demo
results/       CSV, biểu đồ, ảnh và bảng metric
reports/       báo cáo dự án
```

Từ thư mục gốc repo, cài dependencies theo `requirements.txt` và chuẩn bị bộ dữ liệu gốc, rồi:

```bash
python data/io/divide_data.py --source PCB_DATASET --output dataset/pcb_boards_yolo --config configs/data_board.yaml --manifest configs/board_split.json
python tools/train.py --name pcb_v8s_board --model yolov8s.pt --data configs/data_board.yaml --imgsz 1024 --batch 8 --epochs 100 --patience 30 --workers 0 --seed 0 --device 0
python tools/eval.py --weights runs/detect/pcb_v8s_board/weights/best.pt --data configs/data_board.yaml --split val --name pcb_v8s_board --imgsz 1024 --batch 8 --device 0
python tools/analyze_errors.py --weights models/best.pt --data configs/data_board.yaml --split val --conf 0.25 --device cpu --name val_conf_0.25
python tools/select_thresholds.py
python tools/analyze_errors.py --weights models/best.pt --data configs/data_board.yaml --split test --thresholds configs/thresholds.json --device cpu --name test_per_class
python tools/make_report_tables.py
```

Chạy `analyze_errors.py` với `--conf 0.4`, `0.5`, `0.6` (tên `val_conf_0.4`, ...) trước `select_thresholds.py`. Chỉ đánh giá test sau khi chốt mô hình và ngưỡng trên validation. Demo: `python demo/app.py` (cần `models/best.pt` hoặc `HF_MODEL_REPO`). Quy ước Git: mỗi thành viên làm trên nhánh riêng, push và mở pull request; không commit dataset, `runs/` hay `*.pt`.

## 3. Các thí nghiệm

| Run | Cách chia | Mô hình | Kích thước ảnh | Kết quả/ghi chú |
|---|---|---|---:|---|
| `pcb_v8n_random` | Ngẫu nhiên | YOLOv8n | 1024 | Val: P 0,9702; R 0,9333; mAP50 0,9668; mAP50-95 0,5293 |
| `pcb_v8s` | Ngẫu nhiên | YOLOv8s | 1024 | Val: P 0,9728; R 0,9473; mAP50 0,9672; mAP50-95 0,5437 |
| `pcb_v8n_board` | Theo bo | YOLOv8n | 1024 | Val: P 0,9728; R 0,9165; mAP50 0,9722; mAP50-95 0,5195 |
| `pcb_v8s_board` | Theo bo | YOLOv8s | 1024 | Detector chính; có val và test |
| `pcb_best_640` | Theo bo | YOLOv8s | 640 | Val: P 0,9864; R 0,8945; mAP50 0,9552; mAP50-95 0,4963 |
| `pcb_cls_board` | Theo bo | YOLOv8n-cls | 128 | Classifier crop: top-1 val 1,000, test 0,990 (trên box nhãn thật) |

Các lần chạy ghi trong nhật ký ngày 31/10/2026 và 02/11/2026; sửa tên lớp ngày 03/11/2026. Chọn `pcb_v8s_board` dựa trên val theo bo: mAP50-95 0,551 so với 0,519 của v8n, bỏ sót 11 so với 26, đổi lại chậm hơn (24,2 so với 13,3 ms/ảnh). Val chỉ có 2 bo và một seed nên chênh lệch chỉ là gợi ý.

## 4. Phương pháp

**Tiền xử lý.** `data/io/convert_voc_to_yolo.py` đọc tên lớp và tọa độ box từ XML Pascal VOC, giới hạn trong kích thước ảnh và chuyển sang `class_id x_center y_center width height` chuẩn hóa. ID bo lấy từ phần tên ảnh trước dấu gạch dưới; ảnh của cùng một bo vào trọn một tập. Crop cho classifier: vùng vuông quanh lỗi, lề 50%, 128×128, giữ nguyên split theo bo.

**Huấn luyện.** YOLOv8s khởi tạo từ `yolov8s.pt`, `imgsz=1024`, `batch=8`, tối đa 100 epochs, `patience=30`, `seed=0`. Nhãn lớp trong weights từng bị hoán vị so với file nhãn; đã sửa metadata bằng `tools/fix_class_names.py`, không đổi trọng số.

**Hướng XOR.** (1) Ghép ảnh lỗi với ảnh bo chuẩn theo ID bo; (2) căn chỉnh bằng SIFT, lọc tương ứng, homography/RANSAC; (3) nhị phân hóa rồi XOR; (4) loại vùng biên do warp, lọc nhiễu bằng median blur, closing/opening và connected components; (5) lọc theo diện tích, mở rộng box, rồi phân loại crop. Tham số chọn trên val: `close=27`, `min_area=250`, `grow=2.0`.

**Đánh giá.** `tools/eval.py` tính precision, recall, AP theo lớp, mAP và thời gian suy luận. `tools/analyze_errors.py` ghép box dự đoán với nhãn thật ở IoU 0,5 và thống kê đúng, nhầm lớp, bỏ sót, dự đoán thừa.

## 5. Kết quả chính

| Mô hình / phương pháp | Tập | Precision | Recall | mAP50 / F1 | mAP50-95 | Thời gian |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8s board (conf 0,25) | Validation | 0,9752 | 0,9582 | mAP50 0,9782 | 0,5507 | 24,19 ms/ảnh |
| YOLOv8s board (conf 0,25) | Test | 0,8543 | 0,9336 | mAP50 0,9720 | 0,5162 | 25,71 ms/ảnh |
| XOR cải tiến (`new`) | Validation, IoU 0,5 | 0,6204 | 0,6153 | F1 0,6178 | — | — |
| XOR cải tiến (`new`) | Test, IoU 0,5 | 0,6555 | 0,6206 | F1 0,6376 | — | — |

P, R của YOLO là trung bình theo lớp (Ultralytics). Theo lớp trên test:

| Lớp | Precision | Recall |
|---|---:|---:|
| Missing Hole | 0,6272 | 0,9802 |
| Mouse Bite | 0,9509 | 0,9678 |
| Open Circuit | 0,9834 | 0,9600 |
| Short | 0,6130 | 0,9899 |
| Spur | 1,0000 | 0,7138 |
| Spurious Copper | 0,9514 | 0,9900 |

Trên validation, YOLOv8s board có 589 box đúng, 3 nhầm lớp, 11 bỏ sót và 20 dự đoán thừa trên 603 nhãn. Bộ phân loại trên box XOR tìm đúng đạt 342/373 = 0,9169 (accuracy có điều kiện). Classifier crop top-1 0,990 trên test được đo trên crop theo box thật nên là cận trên.

## 6. Phân tích dương tính giả trên test

### 6.1. Hiện tượng

Ở conf 0,25, test có 601 nhãn: 573 đúng, 6 nhầm lớp, 22 bỏ sót và **307 dự đoán thừa** (validation chỉ có 20). Cơ cấu khác nhau: ở val, 18/20 là box lệch vị trí hoặc trùng với lỗi đã ghép; ở test, 299/307 nằm trên vùng nền (IoU với mọi nhãn dưới 0,1). Precision thấp tập trung ở Missing Hole (119 box thừa) và Short (123); P, R từng lớp nằm trong `results/pcb_v8s_board_test/metrics_per_class.csv`.

### 6.2. Các box thừa lặp lại ở vị trí cố định

Gom box thừa theo vị trí tâm (làm tròn 30 px), 290/307 (94,5%) nằm ở 5 vị trí, mỗi vị trí xuất hiện ở 51–60 trong 60 ảnh của cùng một bo; 17 box còn lại rải rác.

| Bo | Lớp bị báo | Vị trí (x, y) | Số ảnh | conf TB | Quan sát trên ảnh | Sau ngưỡng theo lớp |
|---|---|---|---:|---:|---|---|
| 08 | Missing Hole | 180, 1080 | 60 | 0,59 | vòng tròn in giữa hai pad | còn |
| 08 | Missing Hole | 1740, 510 | 59 | 0,34 | vòng tròn tương tự | còn |
| 12 | Short | 1170, 1140 | 60 | 0,63 | nhánh đồng nối giữa hai pad | còn |
| 12 | Short | 1560, 1110 | 60 | 0,39 | chỗ hai đường mạch giao nhau | bị loại |
| 12 | Spurious Copper | 1710, 1560 | 51 | 0,32 | đoạn đồng chéo nằm riêng | còn |

### 6.3. Bằng chứng đây là chi tiết có thật của bo

- Ở bo 08, box `145, 1043, 221, 1119` xuất hiện giống nhau đến từng pixel trong 54/60 ảnh, và có mặt ở 10 ảnh của mỗi lớp lỗi (kể cả `mouse_bite`, `short`, `spur`). Nếu là lỗi bị gán thiếu nhãn thì nó không thể xuất hiện ở ảnh của mọi lớp.
- Cắt đúng vị trí đó từ 6 ảnh test của cùng bo cho nội dung giống hệt nhau (`results/hotspot2_<bo>_<x>_<y>.png`): đây là chi tiết thiết kế của bo, không phải lỗi được thêm vào ảnh.
- Mô hình nhầm các cấu trúc trông giống lỗi: vòng tròn rỗng giống lỗ thiếu, chỗ nối hoặc giao đường mạch giống đoản mạch, đoạn đồng lẻ giống đồng thừa.

Chưa đối chiếu với ảnh bo mẫu `PCB_USED`. Điểm bo 12 (1170, 1140) có thể là thiết kế hợp lệ hoặc một lỗi có sẵn trên bo gốc mà bộ dữ liệu không gán nhãn; chưa phân biệt được từ ảnh.

### 6.4. Vì sao mAP gần như không đổi nhưng precision tụt

Các box thừa có độ tin cậy thấp đến trung bình (trung bình theo lớp 0,31–0,56). AP tính trên toàn đường precision–recall nên ít bị ảnh hưởng (mAP50 0,978 → 0,972), còn precision đo tại một ngưỡng cố định (0,975 → 0,854).

### 6.5. Ngưỡng tin cậy chọn trên validation

Quét conf trên val (P, R gộp tất cả box; "box sai" gồm box thừa và nhầm lớp):

| conf | P | R | F1 | box sai | bỏ sót |
|---|---:|---:|---:|---:|---:|
| 0,25 | 0,962 | 0,977 | 0,970 | 23 | 11 |
| 0,40 | 0,976 | 0,954 | 0,965 | 14 | 25 |
| 0,50 | 0,981 | 0,942 | 0,961 | 11 | 33 |
| 0,60 | 0,987 | 0,905 | 0,945 | 7 | 56 |

Nâng ngưỡng chung từ 0,25 lên 0,40 chỉ bớt 9 box sai nhưng thêm 14 box bỏ sót, nên không có lợi. `tools/select_thresholds.py` chọn ngưỡng từng lớp theo F1 trên val với ràng buộc recall lớp ≥ 0,90; kết quả (`configs/thresholds.json`): **Short 0,5, các lớp còn lại 0,25**. Áp dụng lên test (`analyze_errors.py --thresholds`): đúng 573, nhầm lớp 6, bỏ sót 22 (không đổi), **dự đoán thừa 307 → 245** (Missing Hole 119, Short 61, Spurious Copper 55, Mouse Bite 7, Spur 2, Open Circuit 1). Phần giảm 62 box chính là vị trí Short ở (1560, 1110). Đường F1 của Ultralytics (trung bình theo lớp) đạt đỉnh 0,97 tại conf 0,406 và gần phẳng trong khoảng 0,2–0,6, nên chênh lệch giữa các ngưỡng là nhỏ.

P, R gộp trên test (601 nhãn): trước 573/(573+6+307) ≈ 0,647 và 573/601 ≈ 0,953; sau ngưỡng theo lớp 573/(573+6+245) ≈ 0,695 và 0,953. Đây là cách tính khác với P = 0,854 trung bình theo lớp của Ultralytics, nên không đặt hai số cạnh nhau như cùng một đại lượng.

### 6.6. Phần còn lại và hướng xử lý

245 box thừa còn lại có 229 box nằm ở 4 vị trí trên, với độ tin cậy 0,59–0,63 ở hai vị trí đầu. Ngưỡng chọn trên validation không loại được chúng mà không làm mất box đúng. Các hướng, theo mức hợp lý:

1. **So khớp với bo mẫu (XOR):** vùng giống hệt bo mẫu không khác biệt nên không bị báo; đây là hướng đúng nguyên lý, nhưng cần ảnh bo mẫu (golden board) và căn chỉnh tốt. Chưa chạy bản kết hợp YOLO + XOR nên chưa có số.
2. **Hard negative với hoán đổi bo:** đưa bo 08 và 12 vào train, chọn hai bo khác làm test, rồi đo lại. Không dùng ảnh test để huấn luyện rồi vẫn gọi là kết quả trên bo chưa thấy.
3. **Thêm bo vào huấn luyện:** chữa tận gốc vì mô hình hiện chỉ thấy 6 bo train.

## 7. Kết luận, hạn chế và hướng tiếp theo

Dữ liệu được chuyển nhãn VOC sang YOLO và chia theo ID bo (6/2/2). YOLOv8s (imgsz 1024) đạt mAP50 0,9782 và mAP50-95 0,5507 trên validation; trên test đạt mAP50 0,9720 và mAP50-95 0,5162, recall 0,9336. Precision trên test thấp hơn (0,8543) do 307 box thừa, trong đó 94,5% là các chi tiết cố định của hai bo test bị nhầm thành lỗi. Ngưỡng theo lớp chọn trên validation giảm còn 245 box thừa mà không mất box đúng. Hướng XOR đạt F1 0,6376 trên test ở IoU 0,5, thấp hơn YOLO, nhưng hai hướng đo khác nhau.

Hạn chế:

- Chỉ có 10 bo, một seed, mỗi tập 2 bo; kết quả nhạy với đặc thù từng bo.
- Box thừa trên test chưa được đối chiếu với ảnh bo mẫu; chưa chạy kết hợp YOLO + XOR.
- XOR cần ảnh không lỗi của đúng bo, cùng mặt và góc chụp gần giống; YOLO không cần nhưng dễ nhầm chi tiết thiết kế của bo mới.
- Classifier crop top-1 0,990 đo trên box nhãn thật, không tính lỗi định vị.
- Test đã được dùng để phân tích lỗi; ngưỡng không chọn theo test.

Hướng tiếp theo: thêm bo hoặc đánh giá chéo theo bo, hard negative với hoán đổi bo, kết hợp YOLO với so khớp bo mẫu khi có ảnh mẫu, thêm một seed.

## 8. File kết quả tham khảo

- `configs/train_params.md`, `configs/board_split.json`, `configs/data_board.yaml`, `configs/thresholds.json`.
- `results/summary.csv`, `results/test_metrics.csv`, `results/data_audit.csv`.
- `results/pcb_v8s_board_val/metrics_per_class.csv`, `results/pcb_v8s_board_test/metrics_per_class.csv`.
- `results/pcb_v8s_board_val_errors/`, `results/pcb_v8s_board_test_errors/` (gồm `false_pos_list.csv`), `results/val_conf_*_val_errors/`, `results/test_per_class_test_errors/`.
- `results/hotspot2_*.png`: ảnh cắt các vị trí box thừa.
- `results/pipeline_val.csv`, `results/pipeline_test.csv`, `results/pipeline_test_cls.csv`.
- `reports/bang_so_lieu.md`: bảng tự sinh từ `results/`.

## Tài liệu tham khảo

1. W. Huang, P. Wei, “A PCB Dataset for Defects Detection and Classification,” arXiv:1901.08204, 2019.
2. Ixiaohuihuihui, Tiny-Defect-Detection-for-PCB, GitHub repository.
3. R. Ding et al., “TDD-Net: a tiny defect detection network for printed circuit boards,” *CAAI Transactions on Intelligence Technology*, vol. 4, no. 2, pp. 110–116, 2019.
4. Ultralytics YOLO documentation: https://docs.ultralytics.com
5. OpenCV documentation: https://docs.opencv.org
6. Gradio documentation: https://www.gradio.app/docs