"""PCB defect inspection demo (Gradio 6).

    python demo/app.py                       # local: http://127.0.0.1:7860
    python demo/app.py --device cpu          # no GPU (recommended on a laptop with a small GPU)
    python demo/app.py --host 0.0.0.0 --auth team:secret
Weights: models/best.pt (detector, required) and models/best_cls.pt (crop classifier, needed for the template tabs),
or HF_MODEL_REPO=<user>/<repo>. See demo/README.md.
"""
from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python demo/app.py` as well as `python -m demo.app`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from demo.config import CLASS_DISPLAY, IMAGE_EXTENSIONS, Settings  # noqa: E402
from demo.service import (TABLE_HEADERS, Analysis, AlignmentError, InputError, ModelError, PCBInspector, SOURCE_TEXT,  # noqa: E402
                          detections_table, load_reference_metrics, write_exports)

log = logging.getLogger("pcb.demo")

CSS = """
.verdict h3 { margin: 0 0 4px 0; }
.status-ok { color: #0a7a4b; } .status-bad { color: #b3261e; }
footer { visibility: hidden; }
"""

DISCLAIMER = ("Công cụ phục vụ nghiên cứu và học tập, **không phải hệ thống kiểm định sản xuất**. Mô hình được huấn luyện trên 10 bo "
              "mạch của một bộ dữ liệu công khai; ảnh khác điều kiện chụp có thể cho kết quả kém hơn các số đo trong tab *Thông tin mô hình*.")


# ----------------------------------------------------------------------------------------------------------------------
# presentation helpers
# ----------------------------------------------------------------------------------------------------------------------
def verdict_md(a: Analysis) -> str:
    n = len(a.detections)
    total = a.timings_ms.get("total", 0.0)
    where = {"yolo": "YOLOv8s", "xor": "SIFT + XOR + phân loại", "cross": "YOLOv8s và XOR"}[a.mode]
    if n == 0:
        head = "### Không phát hiện vùng nghi lỗi"
        body = ("Ở ngưỡng hiện tại mô hình không báo vùng nào. Đây **không** phải bảo đảm bo mạch không lỗi: lỗi nhỏ hoặc khó "
                "(đặc biệt Spur) có thể bị bỏ sót; thử hạ ngưỡng tin cậy.")
    else:
        parts = [f"{c} × {name}" for name, c in a.counts().items() if c]
        head = f"### Phát hiện {n} vùng nghi lỗi"
        body = " · ".join(parts)
    return f"{head}\n{body}\n\n<sub>Phương pháp: {where} · {total:.0f} ms · ảnh {a.image_shape[1]}×{a.image_shape[0]} px · mã {a.request_id}</sub>"


def counts_rows(a: Analysis) -> list[list]:
    if a.mode == "cross":
        agg = {"both": 0, "yolo": 0, "xor": 0}
        for d in a.detections:
            agg[d.source] += 1
        return [[SOURCE_TEXT[k], v] for k, v in agg.items()]
    return [[name, c] for name, c in a.counts().items()]


def notes_md(a: Analysis) -> str:
    return "\n".join(f"- {n}" for n in a.notes) if a.notes else ""


def reference_md(inspector: PCBInspector) -> str:
    m = load_reference_metrics(inspector.s.results_dir)
    lines = ["### Kết quả đã đo (tập test: bo 08 và 12, 120 ảnh, 601 nhãn)"]
    if m["test"]:
        t = m["test"][0]
        lines += ["", "**YOLOv8s, imgsz 1024** (mô hình dùng trong demo)", "",
                  "| Precision | Recall | mAP50 | mAP50-95 | ms/ảnh |", "|---:|---:|---:|---:|---:|",
                  f"| {float(t['precision']):.3f} | {float(t['recall']):.3f} | {float(t['AP50']):.3f} | {float(t['AP50-95']):.3f} | "
                  f"{float(t['ms_per_image']):.1f} |"]
    if m["per_class"]:
        lines += ["", "**Theo từng loại lỗi**", "", "| Loại lỗi | Precision | Recall | AP50 | AP50-95 |", "|---|---:|---:|---:|---:|"]
        for r in m["per_class"]:
            if r["class"] != "all":
                lines.append(f"| {r['class']} | {float(r['precision']):.3f} | {float(r['recall']):.3f} | {float(r['AP50']):.3f} | "
                             f"{float(r['AP50-95']):.3f} |")
    xor = [r for r in m["xor"] if r.get("iou") == "0.5"]
    if xor:
        x = xor[0]
        lines += ["", "**Hướng ảnh mẫu (SIFT + XOR), IoU 0,5**", "", "| Precision | Recall | F1 |", "|---:|---:|---:|",
                  f"| {float(x['P']):.3f} | {float(x['R']):.3f} | {float(x['F1']):.3f} |"]
    if not (m["test"] or m["per_class"] or xor):
        lines.append("\n(Không tìm thấy thư mục `results/` cạnh mã nguồn.)")
    return "\n".join(lines)


def model_card_md(inspector: PCBInspector) -> str:
    i = inspector.info()
    det, cls = i["detector"], i["classifier"]
    v = i["versions"]
    return f"""### Thông tin mô hình

| Thành phần | Tệp | Có sẵn | Mã băm (12 ký tự đầu của SHA-256) |
|---|---|---|---|
| Detector YOLOv8s | `{det['file']}` | {'có' if det['available'] else 'không'} | `{det['sha256_12'] or '—'}` |
| Bộ phân loại YOLOv8n-cls (vùng cắt 128×128) | `{cls['file']}` | {'có' if cls['available'] else 'không'} | `{cls['sha256_12'] or '—'}` |

Thiết bị: **{i['device']}** · ultralytics {v['ultralytics'] or '—'} · torch {v['torch'] or '—'} · OpenCV {v['cv2'] or '—'}

Sáu loại lỗi: {', '.join(CLASS_DISPLAY)}.

**Ngưỡng tin cậy mặc định {inspector.s.default_conf:.2f}** được chọn trên tập validation (ưu tiên recall: ít bỏ sót nhất, báo cáo mục 6.5),
không chọn trên test. Điểm của YOLO không phải xác suất đúng: mô hình hiếm khi cho điểm trên 0,8 nên 0,5–0,7 vẫn là kết quả bình thường.

{reference_md(inspector)}

### Giới hạn đã biết
- Trên test, ở ngưỡng 0,25 có 307 dương tính giả (val: 20), phần lớn ở Missing Hole, Short, Spurious Copper và nằm trên nền; nguyên nhân chưa xác định.
- Spur là lớp khó nhất (recall thấp hơn); ở hướng ảnh mẫu, 28/47 vùng Spur trên test bị nhận thành Spurious Copper.
- Hướng ảnh mẫu cần ảnh không lỗi của **đúng bo mạch**, cùng mặt và góc chụp gần giống.
- Chỉ 10 bo, một seed; chưa đánh giá trên ảnh ngoài bộ dữ liệu.

{DISCLAIMER}"""


def discover_examples(directory: Path):
    """Defect images are any image in demo/examples except template_<board>.* ; a defect '<board>_...' pairs with that template."""
    if not directory.is_dir():
        return [], []
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    templates = {p.stem.split("_", 1)[1]: p for p in files if p.stem.lower().startswith("template_")}
    defects = [p for p in files if not p.stem.lower().startswith("template_")]
    pairs = [[str(d), str(templates[d.stem.split("_", 1)[0]])] for d in defects if d.stem.split("_", 1)[0] in templates]
    return [[str(d)] for d in defects], pairs


# ----------------------------------------------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------------------------------------------
def build_ui(inspector: PCBInspector, startup_error: str | None = None) -> gr.Blocks:
    s = inspector.s

    def guarded(fn):
        """Turn the service's user-facing errors into Gradio errors; log anything unexpected with a short id."""
        @functools.wraps(fn)
        def wrapper(*args):
            try:
                return fn(*args)
            except (InputError, ModelError, AlignmentError) as exc:
                raise gr.Error(str(exc), print_exception=False)
            except Exception as exc:  # noqa: BLE001
                log.exception("unexpected error")
                raise gr.Error(f"Lỗi không mong đợi ({type(exc).__name__}). Xem nhật ký máy chủ để biết chi tiết.") from exc
        return wrapper

    def pack(a: Analysis):
        csv_path, json_path = write_exports(a, inspector.info())
        return (a.annotated, verdict_md(a), counts_rows(a), detections_table(a), a.crops, csv_path, json_path, notes_md(a))

    @guarded
    def run_detect(image, conf, imgsz, iou):
        return pack(inspector.detect(image, conf, int(imgsz), iou))

    @guarded
    def run_xor(image, template):
        a = inspector.xor_inspect(image, template)
        return (*pack(a), a.extra_images["xor_map"])

    @guarded
    def run_cross(image, template, conf):
        a = inspector.cross_check(image, template, conf)
        return (*pack(a), a.extra_images["xor_map"])

    defect_examples, pair_examples = discover_examples(s.examples_dir)

    def result_block(prefix: str, with_map: bool = False):
        """Shared output widgets; returns them in the order pack() produces."""
        out_img = gr.Image(label="Kết quả", interactive=False, elem_id=f"{prefix}-img")
        verdict = gr.Markdown(elem_classes="verdict")
        notes = gr.Markdown()
        counts = gr.Dataframe(headers=["Mục", "Số lượng"], interactive=False, label="Tổng hợp")
        table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, wrap=True, label="Chi tiết từng vùng")
        gallery = gr.Gallery(label="Ảnh cận cảnh từng vùng (12 vùng đầu theo độ tin cậy)", columns=4, height="auto", object_fit="contain")
        with gr.Row():
            f_csv = gr.File(label="Tải CSV", interactive=False)
            f_json = gr.File(label="Tải JSON (kèm tham số và phiên bản mô hình)", interactive=False)
        xor_map = gr.Image(label="Bản đồ khác biệt XOR (đỏ = pixel khác ảnh mẫu, đã căn chỉnh)", interactive=False, visible=with_map)
        return out_img, verdict, counts, table, gallery, f_csv, f_json, notes, xor_map

    with gr.Blocks(title=s.title) as app:
        gr.Markdown(f"# {s.title}\nPhát hiện và phân loại 6 loại lỗi trên ảnh bo mạch bằng YOLOv8s, đối chiếu với hướng ảnh mẫu "
                    "(SIFT + XOR + phân loại) theo bài báo của Huang & Wei (2019).")
        if startup_error:
            gr.Markdown(f"<span class='status-bad'>**Chưa sẵn sàng:** {startup_error}</span>")
        else:
            gr.Markdown(f"<span class='status-ok'>Mô hình sẵn sàng · thiết bị {inspector.device}</span>")

        with gr.Tabs():
            with gr.Tab("1 · Phát hiện nhanh (YOLO)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        img = gr.Image(type="filepath", label="Ảnh bo mạch cần kiểm tra", sources=["upload"])
                        conf = gr.Slider(0.05, 0.95, value=s.default_conf, step=0.05, label="Ngưỡng tin cậy",
                                         info="Thấp hơn: thấy nhiều vùng hơn nhưng nhiều báo thừa hơn. Mặc định chọn từ tập validation.")
                        with gr.Accordion("Tùy chọn nâng cao", open=False):
                            imgsz = gr.Radio([640, 800, 1024], value=s.imgsz, label="Kích thước ảnh đưa vào mô hình",
                                             info="Mô hình được huấn luyện và đánh giá ở 1024; 640 nhanh gấp đôi nhưng giảm mAP50-95 khoảng 0,055.")
                            iou = gr.Slider(0.3, 0.9, value=s.iou_nms, step=0.05, label="Ngưỡng IoU khử trùng lặp (NMS)")
                        run1 = gr.Button("Phân tích", variant="primary")
                        if defect_examples:
                            gr.Examples(examples=defect_examples, inputs=[img], label="Ảnh mẫu thử (bo mạch test)")
                    with gr.Column(scale=3):
                        o1 = result_block("t1")
                run1.click(run_detect, [img, conf, imgsz, iou], list(o1[:8]), concurrency_limit=1)

            with gr.Tab("2 · Đối chiếu ảnh mẫu (SIFT + XOR)"):
                gr.Markdown("Cần **ảnh không lỗi của đúng bo mạch đó**. Hệ thống căn chỉnh hai ảnh bằng SIFT, XOR để tìm vùng khác nhau, "
                            "rồi phân loại từng vùng.")
                with gr.Row():
                    with gr.Column(scale=2):
                        img2 = gr.Image(type="filepath", label="Ảnh bo mạch cần kiểm tra", sources=["upload"])
                        tpl2 = gr.Image(type="filepath", label="Ảnh mẫu không lỗi (cùng bo mạch)", sources=["upload"])
                        run2 = gr.Button("Phân tích", variant="primary")
                        if pair_examples:
                            gr.Examples(examples=pair_examples, inputs=[img2, tpl2], label="Cặp ảnh mẫu thử")
                    with gr.Column(scale=3):
                        o2 = result_block("t2", with_map=True)
                run2.click(run_xor, [img2, tpl2], list(o2[:8]) + [o2[8]], concurrency_limit=1)

            with gr.Tab("3 · Đối chiếu hai hướng"):
                gr.Markdown("Chạy cả YOLO và hướng ảnh mẫu trên cùng một cặp ảnh. **Xanh lá**: hai hướng cùng thấy; **xanh dương**: chỉ YOLO; "
                            "**cam**: chỉ XOR. Vùng cả hai cùng thấy đáng tin hơn, vùng chỉ một hướng thấy nên được kiểm tra bằng mắt.")
                with gr.Row():
                    with gr.Column(scale=2):
                        img3 = gr.Image(type="filepath", label="Ảnh bo mạch cần kiểm tra", sources=["upload"])
                        tpl3 = gr.Image(type="filepath", label="Ảnh mẫu không lỗi (cùng bo mạch)", sources=["upload"])
                        conf3 = gr.Slider(0.05, 0.95, value=s.default_conf, step=0.05, label="Ngưỡng tin cậy của YOLO")
                        run3 = gr.Button("Phân tích", variant="primary")
                        if pair_examples:
                            gr.Examples(examples=pair_examples, inputs=[img3, tpl3], label="Cặp ảnh mẫu thử")
                    with gr.Column(scale=3):
                        o3 = result_block("t3", with_map=True)
                run3.click(run_cross, [img3, tpl3, conf3], list(o3[:8]) + [o3[8]], concurrency_limit=1)

            with gr.Tab("Thông tin mô hình"):
                gr.Markdown(model_card_md(inspector))

        gr.Markdown(f"<sub>{DISCLAIMER}</sub>")
    app.queue(max_size=8)
    return app


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link (off by default)")
    ap.add_argument("--auth", help="user:password; strongly advised with --host 0.0.0.0")
    ap.add_argument("--device", help="auto | cpu | 0")
    ap.add_argument("--conf", type=float, help="default confidence threshold")
    ap.add_argument("--det-weights", type=Path)
    ap.add_argument("--cls-weights", type=Path)
    ap.add_argument("--lazy", action="store_true", help="do not load weights at start-up")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def settings_from_args(a: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if a.host:
        s.host = a.host
    if a.port:
        s.port = a.port
    if a.share:
        s.share = True
    if a.device:
        s.device = a.device
    if a.conf:
        s.default_conf = a.conf
    if a.det_weights:
        s.det_weights = a.det_weights
    if a.cls_weights:
        s.cls_weights = a.cls_weights
    if a.auth and ":" in a.auth:
        user, _, password = a.auth.partition(":")
        s.auth = (user, password)
    return s


def main(argv=None) -> None:
    a = parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = settings_from_args(a)
    if settings.host not in ("127.0.0.1", "localhost") and not settings.auth:
        log.warning("serving on %s without authentication; anyone on the network can use this demo", settings.host)
    inspector = PCBInspector(settings)
    startup_error = None
    if not a.lazy:
        try:
            _ = inspector.detector
        except ModelError as exc:
            startup_error = str(exc)
            log.error("detector not ready: %s", exc)
        else:
            try:
                check = inspector.self_check()
            except ModelError as exc:
                log.warning("classifier not ready: %s", exc)
            else:
                if check["ok"] is False:
                    startup_error = (f"Bộ phân loại chỉ đúng {check['correct']}/{check['checked']} ảnh tham chiếu: nhiều khả năng best_cls.pt "
                                     "còn tên lớp bị hoán vị. Chạy: python tools/fix_class_names.py cls <best.pt gốc> models/best_cls.pt")
                    log.error(startup_error)
                elif check["ok"]:
                    log.info("classifier self-check passed: %d/%d", check["correct"], check["checked"])
    app = build_ui(inspector, startup_error)
    app.launch(server_name=settings.host, server_port=settings.port, share=settings.share, auth=settings.auth,
               theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate", font=["system-ui", "Segoe UI", "sans-serif"]),
               css=CSS, show_error=True)


if __name__ == "__main__":
    main()
