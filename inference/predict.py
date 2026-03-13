import argparse
import json
import cv2
from pathlib import Path
from pipeline import PoultryPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the two-stage poultry disease detection pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--stage1", type=str, default="models/stage1/best.pt", help="Path to Stage 1 weights")
    parser.add_argument("--stage2", type=str, default="models/stage2/best.pt", help="Path to Stage 2 weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save annotated output image")
    return parser.parse_args()


def annotate(image_path: str, results: dict, save_path: str):
    image = cv2.imread(image_path)
    for det in results["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = det["diseases"][0]["disease"] if det["diseases"] else "AbNormal"
        cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(save_path, image)
    print(f"Annotated image saved to {save_path}")


def main():
    args = parse_args()
    pipeline = PoultryPipeline(args.stage1, args.stage2, conf=args.conf)
    results = pipeline.run(args.image)

    print(json.dumps(results, indent=2))

    if args.save:
        annotate(args.image, results, args.save)


if __name__ == "__main__":
    main()
