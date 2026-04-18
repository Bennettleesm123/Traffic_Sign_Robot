"""
Merge YOLO detections with HSV traffic-light second stage into a label set for PolicyPipeline.
"""

from __future__ import annotations

import logging
from typing import List, Set, Tuple

import numpy as np

from .detection_filter import resolve_predict_classes
from .settings import RobotSettings
from .traffic_light_color import classify_traffic_light_hsv
from .yolo_inference import run_frame

log = logging.getLogger(__name__)


def _is_generic_traffic_light(label: str, substrings: Tuple[str, ...]) -> bool:
    l = label.lower().replace("-", " ")
    return any(s in l for s in substrings)


def _predict_classes(model, settings: RobotSettings):
    return resolve_predict_classes(model, settings.yolo_class_filter, log_once=True)


def detections_to_policy_labels(
    frame_bgr: np.ndarray,
    details: List[Tuple[str, float, Tuple[int, int, int, int]]],
    settings: RobotSettings,
) -> Set[str]:
    """
    Non–traffic-light detections pass through by class name.
    Generic traffic-light boxes get HSV state labels when possible.
    """
    out: Set[str] = set()
    h, w = frame_bgr.shape[:2]
    for lab, _score, box in details:
        if not _is_generic_traffic_light(lab, settings.traffic_light_generic_substrings):
            out.add(lab)
            continue
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            out.add(lab)
            continue
        crop = frame_bgr[y1:y2, x1:x2]
        area = (y2 - y1) * (x2 - x1)
        if area < settings.min_light_crop_area:
            out.add(lab)
            continue
        state = classify_traffic_light_hsv(crop)
        if state:
            out.add(state)
        else:
            out.add(lab)
    return out


def infer_labels_one_frame(model, frame_bgr: np.ndarray, settings: RobotSettings) -> Set[str]:
    pc = _predict_classes(model, settings)
    _, details = run_frame(
        model,
        frame_bgr,
        conf=settings.conf,
        iou=settings.iou,
        classes=pc,
    )
    return detections_to_policy_labels(frame_bgr, details, settings)


def infer_frame(
    model,
    frame_bgr: np.ndarray,
    settings: RobotSettings,
) -> Tuple[Set[str], List[Tuple[str, float, Tuple[int, int, int, int]]]]:
    """Single YOLO pass; returns policy labels + raw boxes for optional overlay."""
    pc = _predict_classes(model, settings)
    _, details = run_frame(
        model,
        frame_bgr,
        conf=settings.conf,
        iou=settings.iou,
        classes=pc,
    )
    if settings.detection_debug:
        log.debug(
            "infer_frame: predict_classes=%s n_boxes=%s boxes=%s",
            pc,
            len(details),
            [(a, round(b, 2)) for a, b, _ in details],
        )
    labels = detections_to_policy_labels(frame_bgr, details, settings)
    return labels, details
