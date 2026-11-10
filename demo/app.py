"""Demo Gradio. Hai che do:
  - Chi tai anh loi len            -> YOLOv8s (mot giai doan), models/best.pt
  - Tai them anh mau (khong loi)   -> can chinh SIFT + XOR + phan loai, models/best_cls.pt (hai giai doan, theo bai bao)
    python demo/app.py
"""
import os
import sys
from pathlib import Path

import cv2
import gradio as gr
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from pipeline import classify_boxes, draw, xor_detect  # noqa: E402

model_repo = os.getenv("HF_MODEL_REPO")
if model_repo:
    model_path = hf_hub_download(repo_id=model_repo, filename="best.pt")
else:
    model_path = REPO / "models" / "best.pt"
    if not model_path.is_file():
        raise RuntimeError("Set HF_MODEL_REPO to a Hugging Face model repo containing best.pt.")

yolo = YOLO(str(model_path))
cls_path = REPO / "models" / "best_cls.pt"
cls = YOLO(str(cls_path)) if cls_path.exists() else None


def run(defect_path, template_path, conf):
    if defect_path is None:
        return None, []
    if template_path is None:  # mot giai doan
        r = yolo.predict(defect_path, conf=conf, imgsz=1024, verbose=False)[0]
        rows = [[yolo.names[int(c)], round(float(p), 3), int(x1), int(y1), int(x2), int(y2)]
                for (x1, y1, x2, y2), p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        return cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), rows
    if cls is None:
        raise gr.Error("Thieu models/best_cls.pt cho che do XOR.")
    r = xor_detect(template_path, defect_path)  # anh ket qua nam trong toa do anh mau
    preds = classify_boxes(cls, r, r["new"])
    labels = [f"{n} {p:.2f}" for n, p in preds]
    img = draw(r["aligned"], r["new"], labels)
    rows = [[n, round(p, 3), *map(int, b)] for (n, p), b in zip(preds, r["new"])]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rows


demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(type="filepath", label="Anh PCB can kiem tra"),
            gr.Image(type="filepath", label="Anh mau khong loi (tuy chon, bat che do XOR + phan loai)"),
            gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Nguong conf (chi che do YOLO)")],
    outputs=[gr.Image(label="Ket qua"), gr.Dataframe(headers=["Lop", "Do tin cay", "x1", "y1", "x2", "y2"], label="Cac loi")],
    title="Phat hien va phan loai loi tren bang mach in",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
