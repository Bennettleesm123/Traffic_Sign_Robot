"""Discrete hand gestures produced by the rule-based MediaPipe classifier."""

from __future__ import annotations

from enum import Enum, auto


class HandGesture(Enum):
    """Keep in sync with ``gesture_command_map.GESTURE_TO_ROBOT_ACTION``."""

    NONE = auto()  # no stable hand, or unrecognized pose

    # Stop / hold
    OPEN_PALM = auto()  # all fingers spread — natural “halt” in many demos
    FIST = auto()  # closed hand

    # Navigation (assume single hand, selfie camera — see run_robot display / docs for mirroring)
    POINT_LEFT = auto()
    POINT_RIGHT = auto()

    THUMBS_UP = auto()  # “OK / go” — mapped to forward cruise
    THUMBS_DOWN = auto()  # reverse when reliable

    # Optional: victory / peace — mapped to U-turn if enabled in settings
    PEACE = auto()
