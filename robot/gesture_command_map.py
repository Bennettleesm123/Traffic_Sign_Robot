"""
Map ``HandGesture`` → ``RobotAction``. Edit this table to change driving behavior.

Motor wiring (gpiozero ``Robot``): ``left_motor_pins`` / ``right_motor_pins`` in settings correspond to
your H-bridge; label physical M1/M4 in hardware docs — software only knows left vs right wheel.
"""

from __future__ import annotations

from typing import Dict

from .gesture_types import HandGesture
from .sign_policy import RobotAction

# Easy one-place edit: gesture → what the motors should do
GESTURE_TO_ROBOT_ACTION: Dict[HandGesture, RobotAction] = {
    HandGesture.NONE: RobotAction.CRUISE_SLOW,  # idle crawl when no commanding gesture
    HandGesture.OPEN_PALM: RobotAction.STOP,
    HandGesture.FIST: RobotAction.STOP,
    HandGesture.POINT_LEFT: RobotAction.TURN_LEFT,
    HandGesture.POINT_RIGHT: RobotAction.TURN_RIGHT,
    HandGesture.THUMBS_UP: RobotAction.PROCEED,  # forward
    HandGesture.THUMBS_DOWN: RobotAction.REVERSE,
    HandGesture.PEACE: RobotAction.U_TURN,
}
