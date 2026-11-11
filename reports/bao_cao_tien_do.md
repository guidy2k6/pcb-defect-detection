# Báo cáo tiến độ và kết quả đề tài

**Đề tài:** Phát hiện và phân loại lỗi trên bảng mạch in PCB  
**Ngày cập nhật:** 11/11/2026  
**Trạng thái:** Đã có dữ liệu chia theo bo, kết quả YOLO trên validation và test, phân tích lỗi (gồm 307 dương tính giả trên test), ngưỡng tin cậy theo lớp, pipeline XOR, kết hợp YOLO + lọc XOR (precision test 0,609 → 0,966) và demo.

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
| Kết hợp YOLO + lọc XOR | `tools/fusion.py`, `tools/eval_fusion.py` | `configs/fusion.json`, `results/fusion_val_sweep.csv`, `results/fusion_test.csv` |

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
| YOLOv8s thuần (conf theo lớp, P/R gộp, khớp cùng lớp) | Test, IoU 0,5 | 0,609 | 0,953 | F1 0,743 | — | — |
| **YOLOv8s + lọc XOR** (`min_px=10`) | Test, IoU 0,5 | 0,966 | 0,953 | F1 0,960 | — | — |

P, R của YOLO là trung bình theo lớp (Ultralytics). Theo lớp trên test:

| Lớp | Precision | Recall |
|---|---:|---:|
| Missing Hole | 0,6272 | 0,9802 |
| Mouse Bite | 0,9509 | 0,9678 |
| Open Circuit | 0,9834 | 0,9600 |
| Short | 0,6130 | 0,9899 |
| Spur | 1,0000 | 0,7138 |
| Spurious Copper | 0,9514 | 0,9900 |

Trên validation, YOLOv8s board có 589 box đúng, 3 nhầm lớp, 11 bỏ sót và 20 dự đoán thừa trên 603 nhãn. Bộ phân loại trên box XOR tìm đúng đạt 342/373 = 0,9169 (accuracy có điều kiện). Sai số tập trung ở **Spur**: trong 47 vùng Spur tìm đúng, 28 bị gán thành Spurious Copper (recall Spur 19/47 = 0,40), còn các lớp khác đạt từ 0,96 trở lên (`results/pipeline_test_cls.csv`). Hai lớp này đều là phần đồng nhỏ nhô ra cạnh đường mạch nên crop 128×128 rất giống nhau; đây cũng là lý do ưu tiên nhãn lớp từ YOLO (Spur P = 1,00 trên test) khi kết hợp hai hướng. Classifier crop top-1 0,990 trên test được đo trên crop theo box thật nên là cận trên.

### 5.1. So sánh với bài báo gốc

Bài báo [1] đo định vị bằng sai số số lượng Pd = |d − a| / a × 100% (d là số vùng phát hiện, a là số lỗi thật; công thức 9 của bài báo) và phân loại bằng độ chính xác theo lớp. Pd chỉ so **số lượng**, không kiểm tra vị trí, nên ở đây được tính cùng công thức và đặt cạnh P, R theo IoU để thấy đủ hai mặt.

| Phương pháp | Tập | Số dự đoán | Số nhãn | Pd | Ghi chú |
|---|---|---:|---:|---:|---|
| Bài báo [1] (SURF + XOR) | Cả 693 ảnh, không có tập test riêng | 2955 | 2953 | 0,07% (mỗi lớp 0–0,2%) | 1 phát hiện nhầm, 1 vùng trùng |
| XOR cải tiến (nhóm) | Validation | 598 | 603 | 0,83% | F1 0,618 ở IoU 0,5 |
| XOR cải tiến (nhóm) | Test | 569 | 601 | 5,32% | F1 0,638 ở IoU 0,5; recall 0,929 ở IoU 0,1 |
| YOLOv8s thuần | Test | 941 | 601 | 56,57% | P 0,609, R 0,953 |
| YOLOv8s + lọc XOR | Test | 593 | 601 | 1,33% | P 0,966, R 0,953, F1 0,960 |

Phân loại:

| Chỉ số | Bài báo [1] | Nhóm |
|---|---|---|
| Trên ảnh cắt theo nhãn, tập test | 97,74% (thấp nhất Spur 93,65%) | 0,990 (top-1) |
| Trên vùng do bước định vị tìm ra | 99,40% | 0,917 (342/373); Spur 19/47 = 0,40, năm lớp còn lại 0,969–1,000 |
| Thời gian một ảnh | 0,9889 s (đăng ký ảnh 0,6219 s) | YOLO 25,71 ms; đường XOR chưa ghi trong báo cáo |

Cần đọc các số này với các lưu ý sau:

- Bài báo không nói rõ có chia theo bo mạch hay không, và định vị được đánh giá trên toàn bộ ảnh (không có tập giữ lại), còn nhóm đánh giá trên 2 bo chưa thấy khi chọn tham số. Pd chỉ đếm số lượng nên không chứng minh vị trí đúng.
- Đường XOR của nhóm kém bài báo ở Pd (5,32% so với 0,07%) và ở độ chính xác phân loại trên vùng tìm ra, tập trung ở lớp Spur. Nhóm chưa kiểm chứng nguyên nhân; giả thuyết là bộ phân loại huấn luyện trên crop theo nhãn thật nhưng nhận crop từ box XOR (phóng to hệ số 2,0). Bài báo cũng nhận xét crop từ so ảnh mẫu nhỏ hơn crop theo nhãn và ảnh hưởng đến kết quả.
- Nhóm tốt hơn bài báo ở độ chính xác phân loại trên ảnh cắt theo nhãn (0,990 so với 97,74%), nhưng khác thiết lập (kích thước crop, cách chia dữ liệu, bộ phân loại) nên chỉ mang tính tham khảo.
- Thời gian không so trực tiếp được vì khác phần cứng (YOLO đo trên RTX 3050) và khác phạm vi (bài báo gồm đăng ký ảnh).
- Bài báo không đo mAP và không có chỉ số P, R theo IoU, nên các số này chỉ có ở nhóm.

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

1. **So khớp với bo mẫu (XOR) — đã chạy (`tools/fusion.py`, `tools/eval_fusion.py`).** Giữ box YOLO chỉ khi mặt nạ XOR (đã căn chỉnh, lọc nhiễu) có ít nhất `min_px` pixel khác biệt trong box (mở rộng 8 px); box không xác minh được (gần viền đen do warp) được giữ. `min_px = 10` chọn trên validation với ràng buộc recall val giảm không quá 0,01: F1 val 0,9727 → 0,9735, và ổn định với `min_px` từ 10 đến 100 rồi giảm từ 200 (recall tụt, `results/fusion_val_sweep.csv`). Validation gần như không có hotspot (18 → 17 box thừa) nên hiệu quả chủ yếu thể hiện trên test, được chạy đúng một lần (`results/fusion_test.csv`, ngưỡng conf theo lớp, P/R gộp, khớp cùng lớp ở IoU 0,5, 0 ảnh căn chỉnh thất bại):

| Phương pháp | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8s thuần | 573 | 368 | 28 | 0,609 | 0,953 | 0,743 |
| YOLOv8s + lọc XOR | 573 | 20 | 28 | 0,966 | 0,953 | 0,960 |

Loại 348/368 box thừa và không mất box đúng nào (TP giữ nguyên 573). Các box thừa lặp lại không khác ảnh bo mẫu, phù hợp với nhận định ở 6.3 rằng chúng là chi tiết thiết kế của bo. Lưu ý cách tính: số FP của YOLO thuần (368) không đặt cạnh 307/245 ở 6.1 và 6.5 vì cách khớp khác (khớp cùng lớp ở IoU 0,5; nhầm lớp tính là một FP và một FN); tập đánh giá giống nhau (120 ảnh, 601 nhãn).
2. **Hard negative với hoán đổi bo:** đưa bo 08 và 12 vào train, chọn hai bo khác làm test, rồi đo lại. Không dùng ảnh test để huấn luyện rồi vẫn gọi là kết quả trên bo chưa thấy.
3. **Thêm bo vào huấn luyện:** chữa tận gốc vì mô hình hiện chỉ thấy 6 bo train.

## 7. Kết luận, hạn chế và hướng tiếp theo

Dữ liệu được chuyển nhãn VOC sang YOLO và chia theo ID bo (6/2/2). YOLOv8s (imgsz 1024) đạt mAP50 0,9782 và mAP50-95 0,5507 trên validation; trên test đạt mAP50 0,9720 và mAP50-95 0,5162, recall 0,9336. Precision trên test thấp hơn (0,8543) do 307 box thừa, trong đó 94,5% là các chi tiết cố định của hai bo test bị nhầm thành lỗi. Ngưỡng theo lớp chọn trên validation giảm còn 245 box thừa mà không mất box đúng. Hướng XOR đứng riêng đạt F1 0,6376 trên test ở IoU 0,5, thấp hơn YOLO, nhưng hai hướng đo khác nhau. Khi dùng mặt nạ XOR để xác minh box YOLO, precision test tăng từ 0,609 lên 0,966 và F1 từ 0,743 lên 0,960, recall giữ 0,953 (348 box thừa bị loại, không mất box đúng).

Hạn chế:

- Chỉ có 10 bo, một seed, mỗi tập 2 bo; kết quả nhạy với đặc thù từng bo.
- Kết hợp YOLO + XOR cần ảnh bo mẫu không lỗi của đúng bo; chưa đo trường hợp bo mới không có mẫu. `min_px` chọn trên validation gần như không có hotspot nên hiệu quả chủ yếu kiểm chứng trên 2 bo test.
- Lớp Spur còn yếu: recall 0,82 ở detector (18/28 box bỏ sót) và bộ phân loại crop nhầm nhiều sang Spurious Copper.
- XOR cần ảnh không lỗi của đúng bo, cùng mặt và góc chụp gần giống; YOLO không cần nhưng dễ nhầm chi tiết thiết kế của bo mới.
- Classifier crop top-1 0,990 đo trên box nhãn thật, không tính lỗi định vị.
- Test đã được dùng để phân tích lỗi; ngưỡng không chọn theo test.

Hướng tiếp theo: thêm bo hoặc đánh giá chéo theo bo, hard negative với hoán đổi bo, cải thiện recall lớp Spur, thêm một seed.

## 8. File kết quả tham khảo

- `configs/train_params.md`, `configs/board_split.json`, `configs/data_board.yaml`, `configs/thresholds.json`.
- `results/summary.csv`, `results/test_metrics.csv`, `results/data_audit.csv`.
- `results/pcb_v8s_board_val/metrics_per_class.csv`, `results/pcb_v8s_board_test/metrics_per_class.csv`.
- `results/pcb_v8s_board_val_errors/`, `results/pcb_v8s_board_test_errors/` (gồm `false_pos_list.csv`), `results/val_conf_*_val_errors/`, `results/test_per_class_test_errors/`.
- `results/hotspot2_*.png`: ảnh cắt các vị trí box thừa.
- `results/pipeline_val.csv`, `results/pipeline_test.csv`, `results/pipeline_test_cls.csv`.
- `configs/fusion.json`, `results/fusion_val_sweep.csv`, `results/fusion_test.csv`.
- `reports/bang_so_lieu.md`: bảng tự sinh từ `results/`.

## Tài liệu tham khảo

1. W. Huang, P. Wei, “A PCB Dataset for Defects Detection and Classification,” arXiv:1901.08204, 2019.
2. Ixiaohuihuihui, Tiny-Defect-Detection-for-PCB, GitHub repository.
3. R. Ding et al., “TDD-Net: a tiny defect detection network for printed circuit boards,” *CAAI Transactions on Intelligence Technology*, vol. 4, no. 2, pp. 110–116, 2019.
4. Ultralytics YOLO documentation: https://docs.ultralytics.com
5. OpenCV documentation: https://docs.opencv.org
6. Gradio documentation: https://www.gradio.app/docs