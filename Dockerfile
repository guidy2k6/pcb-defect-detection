# CPU image for the demo. Weights are NOT baked in: mount them at /app/models (best.pt, best_cls.pt) or set HF_MODEL_REPO.
#   docker build -t pcb-demo .
#   docker run --rm -p 7860:7860 -v "$PWD/models:/app/models:ro" -e PCB_AUTH=team:secret pcb-demo
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "ultralytics==8.4.194" "gradio>=6.0,<7" huggingface_hub pyyaml

COPY demo ./demo
COPY tools ./tools
COPY configs ./configs
COPY results ./results

ENV PCB_DEVICE=cpu PCB_HOST=0.0.0.0 PCB_PORT=7860 YOLO_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860
USER 1000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/', timeout=3)"
CMD ["python", "demo/app.py"]
