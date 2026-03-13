import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 binary detector (Normal vs AbNormal)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="stage1", help="Run name for output directory")
    return parser.parse_args()


def train(args):
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        plots=True,
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
