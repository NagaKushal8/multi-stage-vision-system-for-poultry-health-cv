"""
Dataset download script for both training stages.

Before running, set your Roboflow API keys as environment variables:

    export ROBOFLOW_API_KEY_1=your_key_here   # Stage 1 dataset
    export ROBOFLOW_API_KEY_2=your_key_here   # Stage 2 dataset

On Windows:
    set ROBOFLOW_API_KEY_1=your_key_here
    set ROBOFLOW_API_KEY_2=your_key_here

Datasets:
    Stage 1 - Disease-Prediction-2 (chicken-disease workspace)
              Binary classification: Normal / AbNormal
              ~140 train / 40 val / 20 test images

    Stage 2 - HenDiseaseDetection-1 (obinson-tobinson workspace)
              Multi-class: Avian Influenza, Fowl Pox, Hen, etc.
              ~246 train / 31 val images
"""

import os
from pathlib import Path
from roboflow import Roboflow


# Both datasets are downloaded into this folder by default
DATA_DIR = Path(__file__).parent


def download_stage1(dest: str = str(DATA_DIR)) -> str:
    """
    Download the Stage 1 binary detection dataset.
    Returns the path to the downloaded dataset folder.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY_1")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY_1 is not set.")

    rf = Roboflow(api_key=api_key)

    # Workspace: chicken-disease | Project: disease-prediction-oryuo | Version: 2
    project = rf.workspace("chicken-disease").project("disease-prediction-oryuo")
    dataset = project.version(2).download("yolov8", location=dest)

    print(f"Stage 1 dataset downloaded to: {dataset.location}")
    return dataset.location


def download_stage2(dest: str = str(DATA_DIR)) -> str:
    """
    Download the Stage 2 fine-grained disease classification dataset.
    Returns the path to the downloaded dataset folder.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY_2")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY_2 is not set.")

    rf = Roboflow(api_key=api_key)

    # Workspace: obinson-tobinson | Project: hendiseasedetection-octdy | Version: 1
    project = rf.workspace("obinson-tobinson").project("hendiseasedetection-octdy")
    dataset = project.version(1).download("yolov8", location=dest)

    print(f"Stage 2 dataset downloaded to: {dataset.location}")
    return dataset.location


if __name__ == "__main__":
    # Running this script directly downloads both datasets into data/
    stage1_path = download_stage1()
    stage2_path = download_stage2()

    print("\nAll datasets ready.")
    print(f"  Stage 1: {stage1_path}")
    print(f"  Stage 2: {stage2_path}")
    print("\nPass these paths to the training scripts:")
    print(f"  python training/train_stage1.py --data {stage1_path}/data.yaml")
    print(f"  python training/train_stage2.py --data {stage2_path}/data.yaml")
