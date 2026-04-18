#!/usr/bin/env python3
"""
Convert LISA ``allAnnotations.csv`` (or equivalent) into Ultralytics YOLO layout.

LISA releases vary slightly; this script supports the pattern used in this repo's legacy
``build_lisa_records.py``: each data row's first comma-separated field is
``path;label;x1;y1;x2;y2;...`` (semicolon-separated).

Steps:
  1. Download LISA from UCSD / Kaggle (see training/DATASETS.md).
  2. Run:  python training/scripts/lisa_csv_to_yolo.py --lisa-root /path/to/lisa --list-tags
  3. Copy training/lisa_class_map.example.json → training/lisa_class_map.json and map tags.
  4. Run conversion (writes YOLO dirs + a generated data.yaml).

Usage:
  python training/scripts/lisa_csv_to_yolo.py --lisa-root ~/data/lisa --annotations allAnnotations.csv \\
      --class-map training/lisa_class_map.json --out ~/data/lisa_yolo
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def _parse_lisa_row(line: str) -> Optional[Tuple[str, str, float, float, float, float]]:
    """Return (relative_image_path, tag, x1, y1, x2, y2) or None."""
    line = line.strip()
    if not line or line.lower().startswith("filename"):
        return None
    # Pattern from build_lisa_records.py: first CSV cell contains semicolon fields
    first_cell = line.split(",")[0]
    parts = first_cell.split(";")
    if len(parts) >= 6:
        image_path, tag, sx, sy, ex, ey = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        try:
            return (
                image_path.strip(),
                tag.strip(),
                float(sx),
                float(sy),
                float(ex),
                float(ey),
            )
        except ValueError:
            pass
    # Fallback: whole line is semicolon-separated
    parts = line.split(";")
    if len(parts) >= 6:
        try:
            return (
                parts[0].strip(),
                parts[1].strip(),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        except ValueError:
            pass
    return None


def collect_tags(annot_path: Path) -> Counter:
    c: Counter = Counter()
    text = annot_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in text[1:]:
        p = _parse_lisa_row(line)
        if p:
            c[p[1]] += 1
    return c


def yolo_line(cls_id: int, x1: float, y1: float, x2: float, y2: float, iw: int, ih: int) -> str:
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0 / iw
    cy = (y1 + y2) / 2.0 / ih
    nw = w / iw
    nh = h / ih
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 1e-6), 1.0)
    nh = min(max(nh, 1e-6), 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="LISA CSV → YOLO dataset")
    ap.add_argument("--lisa-root", required=True, type=Path, help="Root folder containing annotation CSV + image tree")
    ap.add_argument("--annotations", default="allAnnotations.csv", help="CSV filename under lisa-root")
    ap.add_argument("--out", type=Path, default=None, help="Output dataset root (default: lisa-root/../lisa_yolo)")
    ap.add_argument("--class-map", type=Path, help="JSON: LISA tag → rover class name")
    ap.add_argument("--list-tags", action="store_true", help="Print unique tags + counts, then exit")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lisa_root = args.lisa_root.resolve()
    if lisa_root.is_file() and lisa_root.suffix.lower() == ".csv":
        print(
            "Error: --lisa-root must be the FOLDER that contains the CSV, not the CSV file itself.\n"
            f"  You passed: {lisa_root}\n"
            "  Try instead:\n"
            f'    --lisa-root "{lisa_root.parent}" --annotations "{lisa_root.name}"',
            file=sys.stderr,
        )
        sys.exit(1)
    if not lisa_root.is_dir():
        print(
            f"Error: --lisa-root must be an existing directory. Not found: {lisa_root}\n"
            "  Download LISA, unzip it, then point --lisa-root at that folder (the one that contains allAnnotations.csv).",
            file=sys.stderr,
        )
        sys.exit(1)
    annot = lisa_root / args.annotations
    if not annot.is_file():
        print(
            f"Missing annotations file: {annot}\n"
            f"  --lisa-root should be the dataset folder (example: ~/Downloads/lisa).\n"
            f"  Default CSV name is allAnnotations.csv; override with --annotations other.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    tags = collect_tags(annot)
    if args.list_tags:
        print(f"Unique LISA tags ({len(tags)}):")
        for t, n in tags.most_common():
            print(f"  {n:5d}  {t!r}")
        sys.exit(0)

    if not args.class_map or not args.class_map.is_file():
        print("Provide --class-map (JSON). Run with --list-tags first, then edit lisa_class_map.json", file=sys.stderr)
        sys.exit(1)

    raw_map = json.loads(args.class_map.read_text(encoding="utf-8"))
    tag_to_rover: Dict[str, str] = {k: v for k, v in raw_map.items() if not k.startswith("_")}

    rover_classes: List[str] = sorted(set(tag_to_rover.values()))
    rover_to_id: Dict[str, int] = {c: i for i, c in enumerate(rover_classes)}

    out_root = (args.out or lisa_root.parent / "lisa_yolo").resolve()
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # image path -> list of (rover_class, x1,y1,x2,y2)
    boxes_by_image: Dict[str, List[Tuple[str, float, float, float, float]]] = {}
    skipped = 0
    for line in annot.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parsed = _parse_lisa_row(line)
        if not parsed:
            continue
        rel_img, tag, x1, y1, x2, y2 = parsed
        rover = tag_to_rover.get(tag)
        if rover is None:
            skipped += 1
            continue
        boxes_by_image.setdefault(rel_img, []).append((rover, x1, y1, x2, y2))

    rng = random.Random(args.seed)
    items = list(boxes_by_image.items())
    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_ratio)) if len(items) > 1 else 0
    val_set: Set[str] = set(p for p, _ in items[:n_val])

    copied = 0
    for rel_img, boxes in items:
        split = "val" if rel_img in val_set else "train"
        src = lisa_root / rel_img
        if not src.is_file():
            # try relative to common subfolders
            alt = lisa_root / "frames" / rel_img
            src = alt if alt.is_file() else src
        if not src.is_file():
            skipped += 1
            continue
        if Image is None:
            print("Install Pillow: pip install Pillow", file=sys.stderr)
            sys.exit(1)
        with Image.open(src) as im:
            iw, ih = im.size
        stem = Path(rel_img).stem
        # avoid collisions
        dst_name = f"{stem}__{hash(rel_img) % 10_000_000}.jpg"
        dst_img = out_root / "images" / split / dst_name
        shutil.copy2(src, dst_img)

        label_path = out_root / "labels" / split / (Path(dst_name).stem + ".txt")
        lines_out: List[str] = []
        for rover, x1, y1, x2, y2 in boxes:
            cid = rover_to_id[rover]
            lines_out.append(yolo_line(cid, x1, y1, x2, y2, iw, ih))
        label_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
        copied += 1

    yaml_text = "\n".join(
        [
            f"path: {out_root}",
            "train: images/train",
            "val: images/val",
            "nc: " + str(len(rover_classes)),
            "names:",
        ]
        + [f"  {i}: {name}" for i, name in enumerate(rover_classes)]
        + ["",],
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Wrote {out_root} with {copied} images (train/val split). Skipped rows (unmapped tag or missing file): {skipped}")
    print(f"Classes: {rover_classes}")
    print(f"Train with: python -m robot.train_yolo --data {out_root / 'data.yaml'} --model yolov8n.pt --epochs 100")


if __name__ == "__main__":
    main()
