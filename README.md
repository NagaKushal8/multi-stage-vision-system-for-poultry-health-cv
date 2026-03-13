# Hierarchical Poultry Disease Detection Using YOLOv8

A two-stage computer vision pipeline for automated detection and classification of poultry diseases from farm images.

---

## Overview

This project implements a hierarchical detection system that first localizes birds in farm imagery and then classifies specific diseases at the individual bird level. The two-stage approach mirrors real-world farm monitoring constraints where cameras capture wide-angle views and fine-grained disease symptoms need close-up analysis.

---

## Problem Statement

Poultry diseases such as Avian Influenza and Fowl Pox spread rapidly in commercial farms. Manual inspection is time-consuming, inconsistent, and often catches infections too late. A vision-based system capable of scanning flock images and flagging symptomatic birds enables faster intervention and reduces economic losses.

---

## Architecture

```
Farm Image
    |
    v
Stage 1 - Coarse Detection (YOLOv8s)
Binary classification: Normal / AbNormal
    |
    v (AbNormal detections only)
Region Cropping
    |
    v
Stage 2 - Fine Detection (YOLOv8s)
Disease classification: Avian Influenza / Fowl Pox / ...
    |
    v
Final Prediction
```

Stage 1 runs on the full image and flags birds showing abnormal signs. Each flagged region is cropped and passed to Stage 2, which identifies the specific disease. This design reduces false positives in disease labeling by constraining Stage 2 to already-suspicious regions.

---

## Results

### Stage 1 - Binary Detection (Normal vs AbNormal)

| Metric | Value |
|--------|-------|
| mAP50 | 0.991 |
| mAP50-95 | 0.731 |
| Precision | 0.993 |
| Recall | 0.972 |

Per-class breakdown:

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| AbNormal | 0.988 | 0.961 | 0.994 |
| Normal | 0.998 | 0.983 | 0.988 |

### Stage 2 - Disease Classification

| Metric | Value |
|--------|-------|
| mAP50 | 0.518 |
| mAP50-95 | 0.324 |
| Precision | 0.739 |
| Recall | 0.516 |

Per-class breakdown:

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Avian Influenza | 0.718 | 0.290 | 0.298 |
| Fowl Pox | 0.589 | 0.289 | 0.277 |
| Hen | 0.910 | 0.968 | 0.978 |

Stage 2 performance reflects the limited size of the disease-specific dataset (246 training images across 5 classes). The Hen class achieves strong metrics; disease classes require more training data for production use.

Sample outputs are in [`assets/results/`](assets/results/).

---

## Technologies Used

- Python 3.11
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- PyTorch
- [Roboflow](https://roboflow.com) (dataset management)
- OpenCV

---

## Setup

```bash
git clone https://github.com/NagaKushal8/multi-stage-vision-system-for-poultry-health-cv.git
cd multi-stage-vision-system-for-poultry-health-cv
pip install -r requirements.txt
```

---

## Training

### Stage 1

```bash
python training/train_stage1.py --data path/to/Disease-Prediction-2/data.yaml
```

### Stage 2

```bash
python training/train_stage2.py --data path/to/HenDiseaseDetection-1/data.yaml
```

All CLI options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to `data.yaml` |
| `--model` | `yolov8s.pt` | Base weights |
| `--epochs` | 30 / 10 | Training epochs |
| `--imgsz` | 800 | Input image size |
| `--batch` | 16 | Batch size |
| `--name` | `stage1` / `stage2` | Output run name |

---

## Inference

Place trained weights at `models/stage1/best.pt` and `models/stage2/best.pt`, then run:

```bash
python inference/predict.py --image path/to/image.jpg
```

With optional annotated output saved to disk:

```bash
python inference/predict.py --image path/to/image.jpg --save output.jpg
```

Full options:

```
--image    Input image path (required)
--stage1   Path to Stage 1 weights  [default: models/stage1/best.pt]
--stage2   Path to Stage 2 weights  [default: models/stage2/best.pt]
--conf     Confidence threshold     [default: 0.5]
--save     Path to save annotated output image
```

---

## Project Structure

```
.
├── training/
│   ├── train_stage1.py      # Stage 1 training script
│   └── train_stage2.py      # Stage 2 training script
├── inference/
│   ├── pipeline.py          # PoultryPipeline class
│   └── predict.py           # CLI prediction entry point
├── notebooks/
│   └── experimentation.ipynb
├── models/
│   ├── stage1/              # Place stage1 best.pt here
│   └── stage2/              # Place stage2 best.pt here
├── data/
│   └── sample_images/
├── assets/
│   └── results/             # Training plots and prediction samples
├── requirements.txt
└── .gitignore
```

---

## Future Improvements

- Real-time monitoring integration with farm IP cameras
- Mobile deployment via ONNX or TensorRT export
- Larger annotated disease datasets for Stage 2
- Edge AI deployment on NVIDIA Jetson or Raspberry Pi
- Automated alerting system for farm operators
