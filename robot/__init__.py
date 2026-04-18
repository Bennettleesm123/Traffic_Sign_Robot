"""
Raspberry Pi–oriented CV stack: **MediaPipe hand gestures** → gpiozero motors (default).

* ``python -m robot`` or ``python -m robot.run_robot`` — live gesture control.
* ``python -m robot.gesture_camera_test`` — camera + debug HUD (no motors).
* Legacy traffic-sign YOLO path: ``python -m robot.run_sign_robot``.

Motor API: ``robot.sign_policy.RobotAction``, ``robot.motor_controller``.
Gesture mapping table: ``robot.gesture_command_map.GESTURE_TO_ROBOT_ACTION``.
"""

from .gesture_command_map import GESTURE_TO_ROBOT_ACTION
from .gesture_types import HandGesture
from .sign_policy import RobotAction, map_detection_to_action

__all__ = [
    "GESTURE_TO_ROBOT_ACTION",
    "HandGesture",
    "RobotAction",
    "map_detection_to_action",
]
