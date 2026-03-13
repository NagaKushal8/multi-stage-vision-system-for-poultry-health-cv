import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class PoultryPipeline:
    def __init__(self, stage1_weights: str, stage2_weights: str, conf: float = 0.5):
        self.stage1 = YOLO(stage1_weights)
        self.stage2 = YOLO(stage2_weights)
        self.conf = conf

    def run(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        stage1_results = self.stage1(image, conf=self.conf)[0]

        predictions = []
        for box in stage1_results.boxes:
            label = stage1_results.names[int(box.cls)]
            if label.lower() != "abnormal":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            stage2_results = self.stage2(crop, conf=self.conf)[0]

            diseases = []
            for s2_box in stage2_results.boxes:
                disease_label = stage2_results.names[int(s2_box.cls)]
                diseases.append({
                    "disease": disease_label,
                    "confidence": round(float(s2_box.conf), 4),
                })

            predictions.append({
                "bbox": [x1, y1, x2, y2],
                "stage1_label": label,
                "stage1_confidence": round(float(box.conf), 4),
                "diseases": diseases,
            })

        return {
            "image": image_path,
            "detections": predictions,
        }
