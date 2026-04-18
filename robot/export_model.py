"""
Export YOLO weights to ONNX (or TensorRT on device) for faster Raspberry Pi inference.

  python -m robot.export_model --weights path/to/best.pt --onnx best.onnx

Then run the robot with --weights best.onnx --onnx (see run_robot.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Input .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    pt = Path(args.weights)
    model = YOLO(str(pt))
    out = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    if isinstance(out, list):
        out = out[0]
    print(f"Exported ONNX: {out}")


if __name__ == "__main__":
    main()
