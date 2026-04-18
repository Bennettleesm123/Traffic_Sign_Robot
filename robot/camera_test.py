"""
Laptop / desk: **legacy** USB webcam + YOLO object detection (traffic COCO demo).

For **hand-gesture** debugging without YOLO, use ``python -m robot.gesture_camera_test``.

**Camera + detection (default — loads YOLOv8n COCO on first run)**::

    pip install -r robot/requirements-laptop.txt
    python -m robot.camera_test

**Camera only (no inference)**::

    python -m robot.camera_test --no-yolo

**Custom weights (your fine-tuned traffic-sign model)**::

    python -m robot.camera_test --yolo path/to/best.pt --conf 0.25

**Verbose (logs detection counts / labels every 30 frames)**::

    python -m robot.camera_test --verbose

**Stopping / “disconnecting” the camera:** press **q** in the OpenCV window, or use **Ctrl+C** in
the terminal. That exits the program and releases the webcam (no separate disconnect command).

On macOS, if the window does not appear, grant Terminal/Cursor camera access in
System Settings → Privacy & Security → Camera.

Uses the same 720p + MJPEG tuning as ``robot/camera_utils.py`` (Logitech-style USB).

**Why you might see no boxes:** If you previously ran without a model, the HUD showed only FPS.
With defaults fixed, you should see ``yolov8n.pt | N det``. COCO only includes ``stop sign`` and
``traffic light`` among US-style signs; turn/U-turn arrows need a trained checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2

from .camera_utils import configure_logitech_style_capture, is_live_camera_source
from .detection_filter import resolve_predict_classes
from .yolo_inference import (
    COCO_US_SIGN_HINT,
    load_yolo_model,
    model_looks_like_coco80,
    resolve_class_name,
)

log = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test USB camera + YOLO detection on laptop")
    p.add_argument("--source", default="0", help="Camera index (0 = default) or /dev/video0 on Linux")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-mjpeg", action="store_true", help="Disable MJPEG fourcc if image looks wrong")
    p.add_argument(
        "--no-yolo",
        action="store_true",
        help="Skip detection entirely (camera preview only)",
    )
    p.add_argument(
        "--yolo",
        default="yolov8n.pt",
        help="Path to YOLO .pt / .onnx (default: yolov8n.pt COCO; ignored with --no-yolo)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (lower if signs are small / blurry / shot off a monitor)",
    )
    p.add_argument("--verbose", action="store_true", help="Log detection stats every 30 frames")
    p.add_argument(
        "--class-filter",
        choices=("coco_rover", "none"),
        default="coco_rover",
        dest="class_filter",
        help="coco_rover: COCO-80 weights only — person, traffic light, stop sign (no chair/laptop). "
        "none: all classes. Use 'none' with your own fine-tuned weights.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("Could not open camera %s — try --source 1 or check permissions.", args.source)
        sys.exit(1)

    if is_live_camera_source(src):
        configure_logitech_style_capture(
            cap,
            width=args.width,
            height=args.height,
            fps=args.fps,
            prefer_mjpeg=not args.no_mjpeg,
        )

    model = None
    if not args.no_yolo:
        wpath = args.yolo
        log.info("Loading YOLO weights: %s", wpath)
        try:
            model = load_yolo_model(wpath)
        except FileNotFoundError as e:
            log.error("%s", e)
            sys.exit(1)
        except ImportError as e:
            log.error("YOLO needs: pip install ultralytics torch torchvision (%s)", e)
            sys.exit(1)
        except Exception as e:
            log.error("Failed to load model %s: %s", wpath, e)
            sys.exit(1)
        if model_looks_like_coco80(model):
            log.warning(COCO_US_SIGN_HINT)

    win = "camera_test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t0 = time.perf_counter()
    n = 0
    fps_display = 0.0
    warned_no_boxes = False
    pred_classes = None
    if model is not None:
        pred_classes = resolve_predict_classes(model, args.class_filter, log_once=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("Frame grab failed — exiting.")
                break

            n += 1
            if n % 30 == 0:
                elapsed = time.perf_counter() - t0
                fps_display = n / elapsed if elapsed > 0 else 0.0

            stem = Path(args.yolo).name if not args.no_yolo else ""
            if model is None:
                status = "NO MODEL (--no-yolo)"
                annotated = frame
            else:
                pkw = dict(
                    source=frame,
                    conf=args.conf,
                    iou=0.45,
                    max_det=300,
                    verbose=False,
                )
                if pred_classes is not None:
                    pkw["classes"] = pred_classes
                results = model.predict(**pkw)
                r0 = results[0]
                annotated = r0.plot()
                n_det = len(r0.boxes) if r0.boxes is not None else 0
                cf = args.class_filter if not args.no_yolo else "-"
                status = f"{stem} | {n_det} det | filter={cf}"
                if args.verbose and n % 30 == 0:
                    if n_det == 0:
                        log.debug("Frame %d: 0 detections (try --conf 0.15 or hold a real printed sign)", n)
                    else:
                        labs = []
                        if r0.boxes is not None:
                            for j in range(len(r0.boxes)):
                                cid = int(r0.boxes.cls[j].item())
                                confv = float(r0.boxes.conf[j].item())
                                name = resolve_class_name(r0.names, cid)
                                labs.append(f"{name}:{confv:.2f}")
                        log.debug("Frame %d: %s", n, labs)
                if n_det == 0 and n == 90 and not warned_no_boxes:
                    warned_no_boxes = True
                    log.info(
                        "Still 0 detections after ~90 frames — lower --conf, improve lighting, use a "
                        "physical print (not a photo of a screen), or use weights trained on your signs."
                    )

            label = f"FPS ~{fps_display:.1f} | {status} | q=quit"
            cv2.putText(
                annotated,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(win, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
