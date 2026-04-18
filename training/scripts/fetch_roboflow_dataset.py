#!/usr/bin/env python3
"""
Download a Roboflow Universe project as YOLOv8 (Ultralytics) format.

1. Create a free Roboflow account, open your chosen Universe project, note:
     Workspace URL slug, Project slug, Version number.
2. Create an API key (Roboflow dashboard → Account → API).
3. pip install roboflow
4. Run:

     export ROBOFLOW_API_KEY=xxxxxxxx
     python training/scripts/fetch_roboflow_dataset.py \\
         --workspace us-traffic-signs-pwkzx \\
         --project us-road-signs \\
         --version 1 \\
         --out ~/data/roboflow_us_road_signs

See training/DATASETS.md for suggested projects.

After download, open the printed ``data.yaml`` and rename ``names:`` values to match
``robot/sign_policy.py`` if needed, then:

     python -m robot.train_yolo --data ~/data/roboflow_us_road_signs/data.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Directory to download into")
    ap.add_argument("--format", default="yolov8", help="Roboflow export format (default yolov8)")
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("Set ROBOFLOW_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("pip install roboflow", file=sys.stderr)
        sys.exit(1)

    rf = Roboflow(api_key=key)
    proj = rf.workspace(args.workspace).project(args.project)
    ver = proj.version(args.version)
    args.out.mkdir(parents=True, exist_ok=True)
    ver.download(args.format, location=str(args.out))
    print(f"Downloaded to {args.out}. Inspect data.yaml, then run train_yolo.")


if __name__ == "__main__":
    main()
