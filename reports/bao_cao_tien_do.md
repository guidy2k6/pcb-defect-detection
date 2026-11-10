# Báo cáo tiến độ và kế hoạch đề tài

**Đề tài:** Phát hiện và phân loại lỗi trên bảng mạch in PCB  
**Ngày cập nhật:** 10/11/2026  
**Trạng thái:** Đã có dữ liệu chia theo bo, các kết quả YOLO trên validation/test, phân tích lỗi và thử nghiệm pipeline XOR.

## 1. Tổng quan

### 1.1. Hiện trạng và các vấn đề cần lưu ý

Dự án nhận diện sáu loại lỗi: Missing Hole, Mouse Bite, Open Circuit, Short, Spur và Spurious Copper. Theo manifest và nhật ký hiện tại, dữ liệu được chia theo bo thành 453 ảnh train, 120 ảnh validation, 120 ảnh test; số box lần lượt là 1749, 603 và 601. Có 10 ID bo trong split, phân chia 6/2/2. Cách chia này nhằm tránh cùng một bo xuất hiện đồng thời ở train và validation/test.

Các điểm cần trình bày chính xác:

1. **Nguồn dữ liệu:** quy trình trong repo đọc bộ PCB gốc có `images/`, `Annotations/` và `PCB_USED/`. Không nên ghi nhóm dùng bộ VOC_PCB 10.668 ảnh nếu thực tế dùng bộ gốc.
2. **Đã có test độc lập:** `configs/board_split.json` chỉ định các bo test; `results/test_metrics.csv` hiện có kết quả YOLOv8s board trên test. Nhật ký ghi nhận test đã được đánh giá và có một lần chạy lại bằng `--force` sau khi sửa tên lớp; số tổng không đổi. Vì vậy cần mô tả đúng lịch sử, không nói test chỉ chạy đúng một lần tuyệt đối.
3. **Rò rỉ theo bo:** split được thực hiện theo ID bo thay vì chia ngẫu nhiên theo ảnh. `results/data_audit.csv` là báo cáo audit lưu trong repo; cần kiểm tra báo cáo này khi trình bày kết luận về leakage.
4. **Có các mốc so sánh:** repo lưu kết quả YOLOv8n và YOLOv8s, cách chia ngẫu nhiên và theo bo, thêm một thí nghiệm imgsz 640 cùng baseline XOR. Các phương pháp khác nhau cần được so sánh bằng metric phù hợp, không nên xem F1 của XOR và mAP của YOLO là cùng một đại lượng.

### 1.2. Sản phẩm và tình trạng

| Hạng mục | Tình trạng theo workspace hiện tại |
|---|---|
| Mã nguồn và hướng dẫn chạy | Có `README.md`, `requirements.txt`, `configs/`, `data/io/`, `tools/`, `demo/` |
| Dataset YOLO đã chuyển đổi | Không có trong workspace hiện tại; `dataset/` chỉ chứa `.gitkeep` |
| Weights huấn luyện | Không thấy thư mục `runs/` có trong workspace; weights cần có lại để chạy inference/eval từ đầu |
| Kết quả đánh giá và phân tích | Có nhiều CSV, biểu đồ và ảnh trong `results/` |
| Demo | Có `demo/app.py`; demo cần weights cục bộ hoặc cấu hình Hugging Face |
| Slide và báo cáo đầy đủ | Báo cáo tiến độ này được lưu trong `reports/`; không xác nhận có slide trong workspace hiện tại |

### 1.3. Đối chiếu hướng nghiên cứu và cách triển khai

Đề tài triển khai hai hướng. Hướng theo quy trình ảnh mẫu gồm căn chỉnh ảnh lỗi với ảnh PCB chuẩn, chuyển xám/nhị phân, XOR, lọc vùng khác biệt và phân loại vùng lỗi. Hướng YOLO là detector một giai đoạn để làm mốc so sánh và đề xuất bổ sung.

Repo hiện có `tools/baseline_xor.py`, `tools/pipeline.py` và `tools/eval_pipeline.py` cho hướng ảnh mẫu; YOLO dùng `tools/train.py`, `tools/eval.py` và `tools/analyze_errors.py`. Đây là cấu trúc thực tế hiện có. Một số tên trong kế hoạch ban đầu, như `tools/inference.py`, không có trong repo; chức năng demo hiện nằm ở `demo/app.py`.

### 1.4. Quy trình xử lý và file tương ứng

| Khối công việc | Thực hiện trong repo | Đầu ra chính |
|---|---|---|
| Đọc XML, chuyển VOC sang YOLO, chia theo bo | `data/io/divide_data.py`, gọi `convert_voc_to_yolo.py` và `check_xml.py` | `dataset/pcb_boards_yolo/`, `configs/data_board.yaml`, `configs/board_split.json`, `results/data_audit.csv` |
| Huấn luyện detector | `tools/train.py` | `runs/detect/<run>/weights/best.pt`, `last.pt`, `args.yaml`, log và biểu đồ; ghi thông số vào `configs/train_params.md` |
| Sửa tên lớp trong weights khi cần | `tools/fix_class_names.py` | Weights mới tại đường dẫn đích, không train lại |
| Đánh giá detector | `tools/eval.py` | `results/<run>_<split>/metrics_per_class.csv`, `speed.txt`, biểu đồ; test tổng hợp ở `results/test_metrics.csv` |
| Phân tích lỗi | `tools/analyze_errors.py` | Các CSV dưới `results/<run>_<split>_errors/` |
| Tạo ảnh ví dụ lỗi | `tools/error_examples.py` | Ảnh dưới `results/error_examples/` theo nhóm lỗi |
| Tổng hợp các lần eval | `tools/build_summary.py` | `results/summary.csv` |
| Cắt ảnh cho classifier | `data/io/collect_crops.py` | `dataset/cls/{train,val,test}/<class>/*.png` |
| Huấn luyện classifier crop | `tools/train_cls.py` | `runs/classify/<run>/weights/` và kết quả trong `results/` nếu bật đánh giá |
| Đánh giá XOR | `tools/eval_pipeline.py` gọi `tools/pipeline.py` và `tools/baseline_xor.py` | `results/pipeline_<split>.csv`, tùy chọn sweep, overlay và CSV phân loại |

`tools/yolo_utils.py` và `tools/crop_utils.py` là module dùng chung, không chạy độc lập. `tools/eval_baseline.py` đánh giá riêng baseline XOR và có thể ghi `results/baseline_test.csv`.

### 1.5. Nguyên tắc báo cáo kết quả

- Mỗi số liệu cần truy được về một file dưới `results/` hoặc nhật ký `configs/train_params.md`.
- Chọn mô hình và tham số bằng validation; test chỉ dùng báo cáo cuối. Lịch sử chạy lại test cần được nêu minh bạch.
- Ghi tham số mỗi lần chạy; khi so sánh, chỉ thay đổi một yếu tố nếu mục tiêu là đo tác động của yếu tố đó.
- Tách rõ kết quả crop-classifier trên box nhãn thật khỏi kết quả end-to-end có bước tìm box.
- Không so sánh trực tiếp mAP YOLO với F1 XOR như cùng metric; cần báo rõ IoU và cách ghép box.

## 2. Kho mã và hướng dẫn chạy

### 2.1. Cấu trúc repo hiện tại

```text
configs/       cấu hình dữ liệu, manifest chia bo, nhật ký train
data/io/       kiểm tra dữ liệu, chuyển VOC, chia split, tạo crop
tools/         train, eval, phân tích lỗi, pipeline XOR và tiện ích
dataset/       nơi lưu ảnh/nhãn đã chuẩn bị; không có dữ liệu trong workspace hiện tại
models/        nơi đặt weights dùng cho demo
input/         ảnh đầu vào demo dòng lệnh
output/        kết quả demo/baseline
results/       CSV, biểu đồ, ảnh và báo cáo metric
reports/       báo cáo dự án
demo/app.py    giao diện Gradio
```

### 2.2. Trình tự chạy detector từ đầu

Từ thư mục gốc repo, cài dependencies theo `requirements.txt`, rồi chuẩn bị dataset gốc. Sau đó chạy:

```bash
python data/io/divide_data.py --source PCB_DATASET --output dataset/pcb_boards_yolo --config configs/data_board.yaml --manifest configs/board_split.json
python tools/train.py --name pcb_v8s_board --model yolov8s.pt --data configs/data_board.yaml --imgsz 1024 --batch 8 --epochs 100 --patience 30 --workers 0 --seed 0 --device 0
python tools/eval.py --weights runs/detect/pcb_v8s_board/weights/best.pt --data configs/data_board.yaml --split val --name pcb_v8s_board --imgsz 1024 --batch 8 --device 0
python tools/analyze_errors.py --weights runs/detect/pcb_v8s_board/weights/best.pt --data configs/data_board.yaml --split val --name pcb_v8s_board --imgsz 1024 --device cpu
```

Chỉ sau khi chốt mô hình trên validation mới đánh giá test bằng `tools/eval.py --split test`. Nếu không có GPU phù hợp, thay `--device 0` bằng `--device cpu`. Lệnh demo theo README là `yolo predict ...`; giao diện Gradio chạy bằng `python demo/app.py` khi đã có weights.

### 2.3. Cộng tác Git

Quy ước trong kế hoạch là mỗi thành viên làm trên nhánh riêng, commit theo từng phần, push và mở pull request để xem lại trước khi merge. Không commit dataset lớn, thư mục `runs/`, token hoặc secrets. Giữ đường dẫn tương đối trong YAML/code để repo có thể chạy trên máy khác. Trước khi làm theo hướng dẫn này, nhóm cần xác nhận repo remote, quyền truy cập và `.gitignore` thực tế.

### 2.4. Quy ước kết quả

Tên run cần thể hiện mô hình/cách chia; log tham số ghi ở `configs/train_params.md`. File kết quả CSV và hình cần gắn với split cụ thể (`val` hoặc `test`) để người đọc không nhầm số liệu. Không ghi đè `results/test_metrics.csv` nếu không chủ ý chạy lại test.

## 3. Tiến trình thực hiện

### 3.1. Các thí nghiệm đã ghi nhận

Nhật ký `configs/train_params.md` và `results/summary.csv` ghi nhận các run chính sau:

| Run | Cách chia | Mô hình | Kích thước ảnh | Kết quả/ghi chú |
|---|---|---|---:|---|
| `pcb_v8n_random` | Ngẫu nhiên | YOLOv8n | 1024 | Validation: P 0,9702; R 0,9333; mAP50 0,9668; mAP50-95 0,5293 |
| `pcb_v8s` | Ngẫu nhiên | YOLOv8s | 1024 | Validation: P 0,9728; R 0,9473; mAP50 0,9672; mAP50-95 0,5437 |
| `pcb_v8n_board` | Theo bo | YOLOv8n | 1024 | Validation: P 0,9728; R 0,9165; mAP50 0,9722; mAP50-95 0,5195 |
| `pcb_v8s_board` | Theo bo | YOLOv8s | 1024 | Chọn làm detector chính; có validation và test |
| `pcb_best_640` | Theo bo | YOLOv8s | 640 | Validation: P 0,9864; R 0,8945; mAP50 0,9552; mAP50-95 0,4963 |
| `pcb_cls_board` | Theo bo | YOLOv8n-cls | 128 | Classifier crop; nhật ký ghi top-1 val 1,000, test 0,990 |

Các ngày chạy ghi trong nhật ký là 31/10/2026 và 02/11/2026; sửa tên lớp được ghi ngày 03/11/2026. Đây là ngày có trong log repo, khác với lịch dự kiến 15–28/10 trong kế hoạch ban đầu.

### 3.2. Phân tích lỗi detector

`tools/analyze_errors.py` ghép box dự đoán với nhãn thật theo IoU (mặc định 0,5), rồi phân loại thành đúng, nhầm lớp, bỏ sót hoặc dự đoán thừa. Với YOLOv8s board trên validation: 603 box nhãn gồm 589 đúng, 3 nhầm lớp, 11 bỏ sót; có 20 dự đoán thừa. Kết quả được lưu trong `results/pcb_v8s_board_val_errors/`.

### 3.3. Pipeline XOR và classifier

Tham số XOR được chọn trên validation: `close=27`, `min_area=250`, `grow=2.0`. Biến thể `new` đạt F1 0,6178 trên val và 0,6376 trên test tại IoU 0,5. Classifier trên các box tìm đúng đạt 342/373 = 0,9169; đây không phải metric end-to-end. Classifier crop huấn luyện riêng đạt top-1 0,990 trên test theo nhật ký, nhưng dùng crop theo box thật.

## 4. Phương pháp đã thực hiện

### 4.1. Tiền xử lý dữ liệu

Dataset gốc gồm ảnh lỗi và annotation Pascal VOC dạng XML. Script `data/io/convert_voc_to_yolo.py`, được gọi bởi `data/io/divide_data.py`, đọc tên lớp và tọa độ box XML, giới hạn tọa độ trong kích thước ảnh, rồi chuyển mỗi box sang định dạng YOLO đã chuẩn hóa: `class_id x_center y_center width height`.

ID bo được lấy từ phần tên ảnh trước dấu gạch dưới. Các ảnh và nhãn của cùng một bo được đưa trọn vào một split; manifest chia 6 bo train, 2 bo validation, 2 bo test. Quá trình tạo dữ liệu cũng ghi `configs/data_board.yaml`, cập nhật `configs/board_split.json` và tạo báo cáo kiểm tra `results/data_audit.csv`. Theo nhật ký, sau chuyển đổi có 453/120/120 ảnh và 1749/603/601 box tương ứng.

Với thí nghiệm classifier, `data/io/collect_crops.py` đọc box YOLO, cắt vùng vuông quanh từng lỗi, thêm lề 50% và resize về 128×128. Crop giữ nguyên split theo bo của ảnh nguồn để tránh đưa crop cùng bo sang nhiều tập.

### 4.2. Huấn luyện mô hình YOLO

Detector chính được khởi tạo từ trọng số pretrained `yolov8s.pt` và huấn luyện với bộ dữ liệu chia theo bo. Cấu hình ghi trong `configs/train_params.md`: `imgsz=1024`, `batch=8`, tối đa 100 epochs, early stopping `patience=30`, `seed=0`, `workers=0`. Run chính là `pcb_v8s_board`; các run YOLOv8n, chia ngẫu nhiên và imgsz 640 là các mốc so sánh.

`tools/train.py` gọi Ultralytics để train, lưu `best.pt`/`last.pt`, thông số và log dưới `runs/detect/<run>/`, đồng thời ghi thông tin lần chạy vào `configs/train_params.md`. Tên lớp trong weights đã được sửa metadata bằng `tools/fix_class_names.py` khi phát hiện không khớp thứ tự class ID; thao tác đó không thay đổi trọng số học được.

### 4.3. Xử lý ảnh theo hướng XOR

Pipeline ảnh mẫu được triển khai riêng với detector YOLO:

1. Ghép ảnh lỗi với ảnh PCB chuẩn theo ID bo.
2. Chuyển ảnh sang grayscale và tìm đặc trưng để căn chỉnh ảnh lỗi với ảnh chuẩn bằng SIFT, lọc tương ứng và homography/RANSAC.
3. Nhị phân hóa ảnh rồi XOR hai ảnh để làm nổi vùng khác biệt.
4. Loại vùng biên không hợp lệ do phép warp; áp dụng median blur, closing/opening và connected components để lọc nhiễu, nối vùng lỗi và lấy box.
5. Lọc vùng theo diện tích, mở rộng box quanh vùng khác biệt; nếu có classifier thì cắt vùng box và dự đoán lớp.

Logic xử lý nằm ở `tools/baseline_xor.py` và `tools/pipeline.py`; `tools/eval_pipeline.py` đánh giá hai biến thể. Tham số biến thể `new` được chọn trên validation (`close=27`, `min_area=250`, `grow=2.0`). Bộ phân loại crop được train riêng bằng `tools/train_cls.py`; kết quả phân loại crop nhãn thật là cận trên, còn đánh giá trên box XOR tìm được mới phản ánh một phần lỗi của toàn quy trình.

### 4.4. Đánh giá và phân tích lỗi

`tools/eval.py` tính precision, recall, AP theo lớp, mAP và thời gian suy luận trên val/test. Mô hình được chọn bằng validation; test được dùng để báo cáo. `tools/analyze_errors.py` ghép box dự đoán với box thật ở IoU 0,5 để thống kê phát hiện đúng, nhầm lớp, bỏ sót và false positive. `tools/error_examples.py` lưu ảnh minh họa lỗi; `tools/build_summary.py` gom các lần đánh giá thành `results/summary.csv`.

## 5. Kết quả chính

| Mô hình / phương pháp | Tập | Precision | Recall | mAP50 / F1 | mAP50-95 | Thời gian |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8s board | Validation | 0,9752 | 0,9582 | mAP50 0,9782 | 0,5507 | 24,19 ms/ảnh |
| YOLOv8s board | Test | 0,8543 | 0,9336 | mAP50 0,9720 | 0,5162 | 25,71 ms/ảnh |
| XOR cải tiến (`new`) | Validation, IoU 0,5 | 0,6204 | 0,6153 | F1 0,6178 | — | — |
| XOR cải tiến (`new`) | Test, IoU 0,5 | 0,6555 | 0,6206 | F1 0,6376 | — | — |

Trên validation, YOLOv8s board có 589 box đúng, 3 box khớp vị trí nhưng nhầm lớp, 11 box bỏ sót và 20 dự đoán thừa trên 603 nhãn. Bộ phân loại XOR trên các box tìm đúng đạt 342/373 = 0,9169; đây là accuracy có điều kiện, không phải metric end-to-end. Classifier crop riêng đạt top-1 0,990 trên test theo nhật ký, nhưng được đánh giá trên crop theo box nhãn thật.

## 6. Kết luận, hạn chế và hướng tiếp theo

Đã tiền xử lý dữ liệu bằng cách chuyển nhãn VOC sang YOLO, chuẩn hóa tọa độ box và chia dữ liệu theo ID bo theo tỷ lệ 6/2/2. Detector YOLOv8s được fine-tune từ weights pretrained ở kích thước ảnh 1024, batch 8, tối đa 100 epochs và early stopping. Trên validation, mô hình đạt mAP50 0,9782 và mAP50-95 0,5507; trên test đạt mAP50 0,9720 và mAP50-95 0,5162.

Song song, đã triển khai xử lý ảnh theo hướng ảnh mẫu: căn chỉnh, grayscale, nhị phân hóa, XOR, lọc hình thái học và gom vùng thành box; classifier được dùng để gán lớp cho crop. Pipeline XOR cải tiến đạt F1 0,6376 trên test tại IoU 0,5. Kết quả này thấp hơn detector theo các metric báo cáo, nhưng hai hướng có quy trình và cách đo khác nhau.

Giới hạn hiện tại là chỉ có 10 bo và một split theo bo; test đã được dùng để báo cáo; accuracy classifier crop không tính lỗi định vị; workspace hiện không có dataset đã chuyển đổi hoặc weights huấn luyện. Bước tiếp theo là khôi phục dataset/weights để tái lập toàn bộ quy trình, bổ sung dữ liệu từ nhiều bo nếu có, và tiếp tục phân tích false positive trên validation mà không tinh chỉnh theo test.

## 7. File kết quả tham khảo và trạng thái workspace

- `configs/train_params.md`: cấu hình và nhật ký huấn luyện.
- `configs/board_split.json`, `configs/data_board.yaml`: manifest và cấu hình dữ liệu.
- `results/data_audit.csv`: báo cáo audit split/dữ liệu.
- `results/summary.csv`, `results/test_metrics.csv`: kết quả tổng hợp YOLO.
- `results/pcb_v8s_board_val/metrics_per_class.csv`: kết quả theo lớp trên validation.
- `results/pcb_v8s_board_val_errors/`: các bảng phân tích lỗi.
- `results/pipeline_val.csv`, `results/pipeline_test.csv`, `results/pipeline_test_cls.csv`: kết quả pipeline XOR.

Các đường dẫn trên là nơi sinh đầu ra theo cấu hình script. Trong workspace hiện tại, `dataset/` chỉ có `.gitkeep` và không thấy thư mục `runs/`; vì vậy dữ liệu YOLO/crop cùng weights cần được khôi phục từ nguồn lưu trữ trước khi chạy lại từ đầu. Các CSV và biểu đồ trong `results/` vẫn được lưu trong repo.

## Tài liệu tham khảo

1. W. Huang, P. Wei, “A PCB Dataset for Defects Detection and Classification,” arXiv:1901.08204, 2019.
2. Ixiaohuihuihui, Tiny-Defect-Detection-for-PCB, GitHub repository.
3. R. Ding et al., “TDD-Net: a tiny defect detection network for printed circuit boards,” *CAAI Transactions on Intelligence Technology*, vol. 4, no. 2, pp. 110–116, 2019.
4. Ultralytics YOLO documentation: https://docs.ultralytics.com
5. OpenCV documentation: https://docs.opencv.org
6. Gradio documentation: https://www.gradio.app/docs

- **Kết quả validation cao hơn test:** trình bày cả hai, không che giấu; dùng validation để chọn mô hình và giữ test làm đánh giá cuối.
- **Chỉ có 10 bo và một split:** kết luận về khả năng tổng quát còn hạn chế; nên bổ sung bo hoặc nhiều split/seed khi có dữ liệu.
- **False positive trên test:** phân tích sau khi chốt mô hình, không điều chỉnh ngưỡng dựa trên test.
- **Classifier crop cho kết quả cao:** nêu rõ đây là crop theo nhãn thật; báo riêng accuracy trên box do XOR tìm được.
- **Căn chỉnh XOR thất bại:** ghi số ảnh thất bại và cách tính chúng vào thống kê; không bỏ qua các trường hợp lỗi.
- **Thiếu dữ liệu/weights trong workspace:** cần khôi phục dataset và checkpoint phù hợp trước khi chạy lại từ đầu; các kết quả CSV hiện có không thay thế được dữ liệu và weights.
- **Lệch tên lớp:** kiểm tra thứ tự class ID trong nhãn, YAML và weights trước khi diễn giải metric theo lớp.

## 7. Hướng hoàn thiện

1. Khôi phục dataset và weights cần thiết, chạy lại hướng dẫn từ một môi trường sạch để kiểm tra khả năng tái lập.
2. Hoàn thiện demo với detector và, nếu cần, classifier; ghi rõ weights được dùng và cách chạy.
3. Nếu có thêm dữ liệu, đánh giá qua nhiều bo/split để giảm phụ thuộc vào một lần chia.
4. Tiếp tục phân tích false positive và lỗi theo kích thước trên validation; không dùng test để chọn tham số.
5. Hoàn thiện slide và báo cáo cuối, bảo đảm mỗi kết quả liên kết được với CSV/biểu đồ nguồn.

### Tóm tắt

Repo hiện ghi nhận quy trình chia dữ liệu theo bo, nhiều thí nghiệm YOLO, đánh giá test cho YOLOv8s board, phân tích lỗi và pipeline XOR. Kết quả chính của detector được lưu trong `results/test_metrics.csv`; các kết quả validation và phân tích chi tiết nằm dưới `results/`. Khi trình bày, cần phân biệt rõ số liệu validation/test, detector/classifier crop và metric YOLO/XOR; đồng thời nêu rằng dataset và weights không có trong workspace hiện tại.

## Tài liệu tham khảo

1. W. Huang, P. Wei, “A PCB Dataset for Defects Detection and Classification,” arXiv:1901.08204, 2019.
2. Ixiaohuihuihui, Tiny-Defect-Detection-for-PCB, GitHub repository.
3. R. Ding et al., “TDD-Net: a tiny defect detection network for printed circuit boards,” *CAAI Transactions on Intelligence Technology*, vol. 4, no. 2, pp. 110–116, 2019.
4. Ultralytics YOLO documentation: https://docs.ultralytics.com
5. OpenCV documentation: https://docs.opencv.org
6. Gradio documentation: https://www.gradio.app/docs