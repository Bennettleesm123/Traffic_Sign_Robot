"""
YOLO class filtering at inference time.

COCO (80-class) checkpoints label chairs, laptops, etc. Ultralytics supports ``predict(..., classes=[...])``
to run NMS only on selected class indices — this removes those objects entirely (not just hiding boxes).

Rover default for COCO weights: keep **person** (0), **traffic light** (9), **stop sign** (11).
This does **not** add left/right/U-turn signs; those classes do not exist in COCO — you need a fine-tuned model
(see ``training/rover_signs.yaml.example``).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .yolo_inference import model_looks_like_coco80

log = logging.getLogger(__name__)

# Ultralytics / COCO 80 order (same as YOLOv5/v8 default coco.yaml)
COCO80_PERSON = 0
COCO80_TRAFFIC_LIGHT = 9
COCO80_STOP_SIGN = 11

COCO_ROVER_ALLOWED_INDICES: tuple[int, ...] = (
    COCO80_PERSON,
    COCO80_TRAFFIC_LIGHT,
    COCO80_STOP_SIGN,
)

_filter_logged = False


def resolve_predict_classes(
    model,
    filter_mode: str,
    *,
    log_once: bool = True,
) -> Optional[List[int]]:
    """
    Returns ``classes=`` list for ``model.predict``, or ``None`` for all classes.

    * ``none`` — no filter.
    * ``coco_rover`` — if the loaded model looks like COCO-80, restrict to person + traffic light + stop sign.
      For custom-trained weights, returns ``None`` (indices would not match your dataset).
    """
    global _filter_logged
    mode = (filter_mode or "none").strip().lower()
    if mode in ("none", "off", "all"):
        return None
    if mode in ("coco_rover", "coco", "rover"):
        if model_looks_like_coco80(model):
            if log_once and not _filter_logged:
                log.info(
                    "YOLO class filter=coco_rover: indices %s → person, traffic light, stop sign only",
                    list(COCO_ROVER_ALLOWED_INDICES),
                )
                _filter_logged = True
            return list(COCO_ROVER_ALLOWED_INDICES)
        if log_once and not _filter_logged:
            log.info(
                "YOLO class filter=coco_rover ignored: weights are not COCO-80 (custom model — using all classes)",
            )
            _filter_logged = True
        return None
    raise ValueError(f"Unknown yolo_class_filter mode: {filter_mode!r}")
