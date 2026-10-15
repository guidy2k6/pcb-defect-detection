![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-Editor-007ACC?logo=visualstudiocode&logoColor=white)

# PCB-Defect-Detection
A computer vision project detecting defects in printed circuit boards (PCBs) using basic vision techniques and yolo8n

## Note

This project is inspired by and builds upon the research presented in the following paper:
[https://huggingface.co/papers/1901.08204](https://huggingface.co/papers/1901.08204)
<br> We would like to thank the authors for making their research publicly available. Their work provides a valuable foundation for our project. While our implementation follows the original research in several aspects, we introduce some modifications and explore its application to additional problems.

## Dataset
We also found a dataset in another repository that is relevant to our research topic. It may serve as a useful resource for experimentation and further development.
<br> You can download the dataset from: [https://pan.baidu.com/s/1eAxDF4txpgMInxbmNDX0Zw](https://pan.baidu.com/s/1eAxDF4txpgMInxbmNDX0Zw) *extraction code:* `a6rh`
<br>Please refer to the original source for further information about the dataset, including its contents, usage conditions, and licensing terms.

## Features

- Automatic PCB defect detection
- YOLOv8n object detection model
- Multiple defect classification
- Bounding box visualization
- Confidence score prediction
- Fast inference
- Computer Vision based inspection

## Defect Classes

The model detects six PCB defects:

| Class |
|--------|
| Mouse Bite |
| Spur |
| Missing Hole |
| Short |
| Open Circuit |
| Spurious Copper |

## Development Environment

- Python
- YOLOv8n
- Ultralytics
- OpenCV
- NumPy
- VS Code
- Jupiter




## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/PCB-Defect-Detection.git
cd PCB-Defect-Detection
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```
The exact package versions can be recorded using:

```bash
pip freeze > requirements.txt
```
## Inference Demo

To run inference on a directory of input images:
```bash
python tools/inference.py --data_dir input --save_dir output
```
--data_dir: Directory containing input images.

--save_dir: Directory where prediction results will be saved.

Make sure the trained model weights are available and that the script arguments match the actual implementation.

## Training and Evaluation

#### Training

Run the training script:

```bash
python tools/train.py
```

Training parameters, including the dataset configuration, model weights, image size, batch size, and number of epochs, should be configured in the script or passed as command-line arguments.

#### Evaluation

Run the evaluation script:
```bash
python tools/eval.py
```
<br> The model is evaluated using the following metrics:
- **Precision**: Proportion of predicted defects that are correct.
- **Recall**: Proportion of actual defects successfully detected.
- **Average Precision (AP)**: Detection performance for each defect category.
- **Mean Average Precision (mAP)**: Overall detection performance across categories.
<br> Results should be reported for each defect category as well as for the overall model.

## Results

<br>This section will contain sample prediction images and Precision–Recall (PR) curves generated during evaluation.
- Prediction examples: Add images from the results/ directory.
- PR curves: Add the evaluation plots from the results/ directory.
- Quantitative results: Add precision, recall, AP, and mAP values after evaluation.
<br>No performance results are reported until the experiments have been completed.

## Acknowledgements

<br> We would like to thank the authors of the original research for making their work publicly available. Their research provides a valuable foundation for this project.
<br> Our implementation follows the original work in several aspects while introducing modifications to explore its applicability to additional problems.
<br> We encourage readers to consult the original paper and reference repository for further details.
