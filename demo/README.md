# Demo: PCB defect inspection

Gradio app with three analyses and a model card.

| Tab | Input | Method |
|---|---|---|
| 1. Phát hiện nhanh | one image | YOLOv8s detector (imgsz 1024) |
| 2. Đối chiếu ảnh mẫu | image + defect-free image of the same board | SIFT alignment, XOR, crop classifier (the paper's route) |
| 3. Đối chiếu hai hướng | image + template | both of the above, boxes matched at IoU ≥ 0.1 (green: both, blue: YOLO only, orange: XOR only) |
| Thông tin mô hình | - | weight file hashes, versions, measured test results, known limits |

Every analysis returns the annotated image, a summary, a per-region table, close-up crops, and a CSV and JSON download (the JSON records
parameters, timings, device and weight hashes).

## Run
```bash
pip install -r requirements.txt
python demo/app.py                    # http://127.0.0.1:7860
python demo/app.py --device cpu       # recommended on a laptop GPU with 6 GB or less
python demo/app.py --host 0.0.0.0 --auth team:secret
```
Weights go in `models/`: `best.pt` (detector, required) and `best_cls.pt` (classifier, needed for tabs 2 and 3). Both must have been
patched with `tools/fix_class_names.py` (see `configs/train_params.md`, entry 03/11/2026). Alternatively set `HF_MODEL_REPO=<user>/<repo>`.

| Setting | Flag | Environment variable | Default |
|---|---|---|---|
| detector weights | `--det-weights` | `PCB_DET_WEIGHTS` | `models/best.pt` |
| classifier weights | `--cls-weights` | `PCB_CLS_WEIGHTS` | `models/best_cls.pt` |
| device | `--device` | `PCB_DEVICE` | `auto` |
| default confidence | `--conf` | `PCB_CONF` | 0.25 (best recall on val, see report section 6.5) |
| host / port | `--host` `--port` | `PCB_HOST` `PCB_PORT` | `127.0.0.1` / `7860` |
| basic auth | `--auth user:pass` | `PCB_AUTH` | off |
| upload limit | - | `PCB_MAX_UPLOAD_MB` | 30 |

## Safeguards
- **Class-name guard.** Detector weights that still carry the old permuted names are refused with the command that fixes them.
- **Classifier self-check.** On start the demo classifies the reference crops in `demo/assets/canary/` and shows a red banner if most are wrong
  (an unpatched `best_cls.pt` does this, and a names check cannot see it). Create the crops once with
  `python tools/make_canary.py --split val` and commit them (12 small PNGs).
- **Input validation.** Type, size, decodability and minimum size are checked; non-ASCII file names are handled.
- **GPU out-of-memory** falls back to CPU once and says so in the result.
- **Alignment failures** (wrong template, no common features, implausible homography) give a message instead of a stack trace.
- Binding to a non-local host without `--auth` logs a warning.

## Example images
Put a few test-board images in `demo/examples/` named like the dataset (`08_short_01.jpg`) plus `template_08.jpg` (the board's file from
`PCB_USED`). They appear as clickable examples; defects pair automatically with the template of the same board number.

## Tests
```bash
python -m pytest -q tests      # fake models: needs numpy, opencv, pytest, pyyaml (gradio optional); no weights, no GPU
```
Covers input validation, name guards, sorting and exports, OOM fallback, the template route on a synthetic board, the cross-check and the
classifier self-check.

## Docker (CPU)
See the header of `Dockerfile`. Weights are mounted, not baked into the image.

## Limits
Research and teaching tool, not a production inspection system. Measured results and known limits are on the *Thông tin mô hình* tab.
The cross-check rule (tab 3) is illustrative and has not been evaluated as a method of its own. Tab 4 (YOLO + XOR filter, `tools/fusion.py`) is the evaluated fusion: results are in `results/fusion_test.csv` and the model-info tab.