"""
Train a YOLOv8 detector on a custom dataset in YOLO-format (images + labels).

Prepare ``data.yaml`` (see ``training/rover_finetune.yaml``, ``training/rover_signs.yaml.example``).

Dataset sources & conversion: **``training/DATASETS.md``** (LISA, Roboflow, Mapillary).
Scripts: ``training/scripts/lisa_csv_to_yolo.py``, ``training/scripts/fetch_roboflow_dataset.py``.

Run:

  python -m robot.train_yolo --data training/rover_finetune.yaml --epochs 100

Weights appear under ``runs/detect/train/weights/best.pt`` by default (Ultralytics).
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="Base checkpoint")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
