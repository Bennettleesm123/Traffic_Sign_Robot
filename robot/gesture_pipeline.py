"""
Glue layer: ``GestureRecognizer`` + ``GestureStabilizer`` → ``RobotAction``.

Keeps vision, temporal filtering, and motor-side policy separate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .gesture_command_map import GESTURE_TO_ROBOT_ACTION
from .gesture_recognition import GestureRecognizer
from .gesture_smoothing import GestureStabilizer, StableReading
from .gesture_types import HandGesture
from .sign_policy import RobotAction


def stable_to_action(stable: Optional[StableReading]) -> Optional[RobotAction]:
    """None → caller should hold previous motor command; otherwise use mapped action."""
    if stable is None:
        return None
    return GESTURE_TO_ROBOT_ACTION.get(stable.gesture, RobotAction.CRUISE_SLOW)


def process_frame_raw(
    recognizer: GestureRecognizer,
    frame_bgr: np.ndarray,
) -> Tuple[HandGesture, float]:
    """Single frame classification (no smoothing)."""
    return recognizer.process_frame(frame_bgr)
