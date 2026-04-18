#!/usr/bin/env python3
"""
Create a rover-sign YOLO dataset layout from a folder of images.

YOLO training needs, per image, a sibling under labels/ with the same stem and ``.txt`` extension.
Each line: ``class_id cx cy w h`` (normalized 0–1). Class ids must match ``data.yaml`` (see below).

* Raw images only: copies images and creates **empty** label files so the folder is valid. You must edit
  those ``.txt`` files (or use Roboflow / Label Studio / labelImg) before training means anything.
* If you already have YOLO labels: pass ``--labels-dir`` with matching ``.txt`` files (same stems as images).

Writes ``data.yaml`` at ``--out`` with an absolute ``path`` so training works from any cwd.

Example:

  python training/scripts/bootstrap_rover_dataset.py --images ~/Pictures/rover_captures
  # label the .txt files (class 0–4), then:
  python -m robot.train_yolo --data datasets/rover_signs/data.yaml --model yolov8n.pt --epochs 100 --batch 8
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Must match training/rover_finetune.yaml and robot/sign_policy.DEFAULT_LABEL_TO_ACTION strings.
CLASS_NAMES = (
    "stop sign",
    "left turn only",
    "right turn only",
    "u turn only",
    "traffic light",
)


def _images_in(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMG_EXT:
            out.append(p)
    return out


def _write_data_yaml(out_root: Path) -> None:
    lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    lines.append("")
    (out_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap YOLO dataset for rover signs")
    ap.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Folder containing your image files (.jpg, .png, …)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/rover_signs"),
        help="Dataset root to create (default: datasets/rover_signs)",
    )
    ap.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional folder of YOLO label .txt files (same stem as each image)",
    )
    ap.add_argument("--val-fraction", type=float, default=0.15, help="Fraction for val split (default 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-empty-labels",
        action="store_true",
        help="Do not create empty label files; require --labels-dir with a .txt per image",
    )
    args = ap.parse_args()

    img_dir: Path = args.images
    out_root: Path = args.out
    if not img_dir.is_dir():
        print(f"Not a directory: {img_dir}", file=sys.stderr)
        sys.exit(1)
    if not (0.0 < args.val_fraction < 0.5):
        print("--val-fraction should be between 0 and 0.5", file=sys.stderr)
        sys.exit(1)

    imgs = _images_in(img_dir)
    if not imgs:
        print(f"No images found in {img_dir} (supported: {sorted(_IMG_EXT)})", file=sys.stderr)
        sys.exit(1)

    rnd = random.Random(args.seed)
    rnd.shuffle(imgs)
    if len(imgs) < 2:
        val_set: set[Path] = set()
    else:
        n_val = max(1, int(len(imgs) * args.val_fraction))
        n_val = min(n_val, len(imgs) - 1)
        val_set = set(imgs[:n_val])
    labels_src: Path | None = args.labels_dir

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for src in imgs:
        split = "val" if src in val_set else "train"
        stem = src.stem
        ext = src.suffix
        dst_img = out_root / "images" / split / f"{stem}{ext}"
        dst_lbl = out_root / "labels" / split / f"{stem}.txt"

        txt_src: Path | None = None
        if labels_src is not None:
            for cand in (labels_src / f"{stem}.txt", labels_src / f"{stem}.TXT"):
                if cand.is_file():
                    txt_src = cand
                    break
        if args.no_empty_labels:
            if txt_src is None:
                print(f"Skipping (no label): {src.name}", file=sys.stderr)
                skipped += 1
                continue
        shutil.copy2(src, dst_img)
        copied += 1
        if txt_src is not None:
            shutil.copy2(txt_src, dst_lbl)
        elif not args.no_empty_labels:
            dst_lbl.write_text("", encoding="utf-8")

    _write_data_yaml(out_root)

    if labels_src is None and not args.no_empty_labels:
        print(
            "WARNING: Created empty label .txt files. Add YOLO lines (class cx cy w h) before training, "
            "or training will not learn signs.\n"
            "Class indices: "
            + ", ".join(f"{i}={name!r}" for i, name in enumerate(CLASS_NAMES)),
            file=sys.stderr,
        )

    print(f"Wrote {copied} images to {out_root} (skipped {skipped}). yaml: {out_root / 'data.yaml'}")
    print(
        "Train with:\n"
        f"  python -m robot.train_yolo --data {(out_root / 'data.yaml').resolve()} "
        "--model yolov8n.pt --epochs 100 --batch 8"
    )


if __name__ == "__main__":
    main()
