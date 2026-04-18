"""
Optional Ultralytics YOLOv8 inference helper for Raspberry Pi.

Train or download a `.pt` or export to TensorRT / ONNX for faster Pi inference:
  yolo export model=best.pt format=onnx

Usage:
  python -m robot.yolo_inference --weights path/to/best.pt --source 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

_WEIGHTS_HINT = (
    "Train first (creates runs/detect/.../weights/best.pt), e.g.:\n"
    "  python -m robot.train_yolo --data path/to/data.yaml --model yolov8n.pt --epochs 100\n"
    "Or use pretrained COCO without training:\n"
    "  python -m robot.camera_test --yolo yolov8n.pt"
)


def validate_local_weights(weights: str) -> None:
    """
    Fail fast with a clear message if a filesystem path to weights does not exist.

    Lets bare names like ``yolov8n.pt`` through so Ultralytics can download from the hub.
    """
    w = weights.strip()
    if w.startswith("http://") or w.startswith("https://"):
        return
    p = Path(w).expanduser()
    if p.is_file():
        return
    # Single path component, relative: hub name or cwd (YOLO resolves / downloads)
    if len(p.parts) == 1 and not p.is_absolute():
        return
    raise FileNotFoundError(
        f"Weights file not found: {p.resolve()}\n\n{_WEIGHTS_HINT}"
    )

def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Install ultralytics: pip install ultralytics (see robot/requirements-pi.txt)"
        ) from e
    return YOLO


def load_yolo_model(weights: str):
    """Load `.pt`, `.onnx`, `.engine`, etc. (Ultralytics-supported)."""
    validate_local_weights(weights)
    YOLO = _load_yolo()
    return YOLO(weights)


def resolve_class_name(names_map, class_id: int) -> str:
    """Ultralytics may expose ``names`` as dict[int,str] or list-like; keys must match ``int``."""
    if isinstance(names_map, dict):
        if class_id in names_map:
            return str(names_map[class_id])
        s = str(class_id)
        if s in names_map:
            return str(names_map[s])
    if isinstance(names_map, (list, tuple)) and 0 <= class_id < len(names_map):
        return str(names_map[class_id])
    return str(class_id)


def model_looks_like_coco80(model) -> bool:
    """Heuristic: default yolov8n.pt has 80 COCO classes including stop sign / traffic light."""
    nmap = getattr(model, "names", None)
    if not isinstance(nmap, dict) or len(nmap) != 80:
        return False
    vals = {str(v).lower() for v in nmap.values()}
    return "stop sign" in vals and "traffic light" in vals


# Classes COCO actually knows (for user expectations). Left/right/U-turn regulatory signs are NOT here.
COCO_US_SIGN_HINT = (
    "COCO-pretrained YOLO (e.g. yolov8n.pt) can detect 'stop sign' and 'traffic light' only among "
    "your rover sign set. 'Left turn only', 'right turn only', and U-turn signs are NOT COCO classes — "
    "train a custom model (see training/rover_signs.yaml.example + training/CV_Rover_Signs.pdf). "
    "With default --class-filter coco_rover, inference uses classes [0,9,11] only (person, traffic light, stop sign) "
    "so chairs/laptops/etc. are suppressed."
)


def run_frame(
    model,
    frame_bgr,
    conf: float = 0.35,
    iou: float = 0.45,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
) -> Tuple[Set[str], List[Tuple[str, float, Tuple[int, int, int, int]]]]:
    """
    Returns (unique class names in frame, list of (name, score, (x1,y1,x2,y2))).
    ``frame_bgr`` is OpenCV BGR; Ultralytics handles conversion internally.
    ``classes`` — optional list of class indices (COCO or your dataset) to keep; ``None`` = all classes.
    """
    pred_kw = dict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )
    if classes is not None:
        pred_kw["classes"] = classes
    results = model.predict(**pred_kw)
    r0 = results[0]
    out_names: Set[str] = set()
    details: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    if r0.boxes is None or len(r0.boxes) == 0:
        return out_names, details
    xyxy = r0.boxes.xyxy.cpu().numpy()
    cls = r0.boxes.cls.cpu().numpy()
    sc = r0.boxes.conf.cpu().numpy()
    id_to_name = r0.names
    for i in range(len(xyxy)):
        cid = int(cls[i])
        label = resolve_class_name(id_to_name, cid)
        out_names.add(label)
        box = tuple(int(x) for x in xyxy[i])
        details.append((label, float(sc[i]), box))
    return out_names, details


def main(argv: Sequence[str] | None = None) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to YOLO .pt (or exported engine/onnx)")
    ap.add_argument(
        "--source",
        default="0",
        help="Webcam index or video path (default 0)",
    )
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--width", type=int, default=1280, help="Webcam width (720p default)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-mjpeg", action="store_true")
    ap.add_argument(
        "--class-filter",
        choices=("coco_rover", "none"),
        default="coco_rover",
        dest="class_filter",
        help="coco_rover: only person, traffic light, stop sign on COCO-80 weights",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    import cv2

    from .camera_utils import configure_logitech_style_capture, is_live_camera_source
    from .detection_filter import resolve_predict_classes

    model = load_yolo_model(args.weights)
    pcls = resolve_predict_classes(model, args.class_filter, log_once=True)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if cap.isOpened() and is_live_camera_source(src):
        configure_logitech_style_capture(
            cap,
            width=args.width,
            height=args.height,
            fps=args.fps,
            prefer_mjpeg=not args.no_mjpeg,
        )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        names, details = run_frame(model, frame, conf=args.conf, classes=pcls)
        for lab, sc, (x1, y1, x2, y2) in details:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{lab} {sc:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        cv2.imshow("yolo", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
