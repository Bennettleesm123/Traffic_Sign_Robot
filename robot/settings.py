"""
Central tunables for vision, gesture smoothing, and GPIO. Override via CLI in run_robot.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RobotSettings:
    # --- Legacy YOLO sign pipeline (run_sign_robot only) ---
    weights: str = "yolov8n.pt"
    source: str = "0"
    conf: float = 0.35
    iou: float = 0.45
    yolo_class_filter: str = "coco_rover"
    detection_debug: bool = False

    # USB webcam (e.g. Logitech 720p)
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: float = 30.0
    camera_prefer_mjpeg: bool = True

    # Legacy traffic-light HSV (run_sign_robot / vision_pipeline)
    traffic_light_generic_substrings: Tuple[str, ...] = (
        "traffic light",
        "traffic_light",
        "trafficlight",
        "tl",
        "signal",
    )
    min_light_crop_area: int = 400

    # Legacy sign policy timings (run_sign_robot)
    stop_hold_seconds: float = 3.0
    yellow_pause_seconds: float = 1.5
    uturn_cooldown_seconds: float = 6.0

    # --- Hand gestures (run_robot / gesture_camera_test) ---
    # MediaPipe Hands — keep detection reasonably high so we don't chase noise on Pi camera feeds.
    gesture_min_detection_conf: float = 0.65
    gesture_min_tracking_conf: float = 0.5
    # Flip processing so “point left” matches a mirrored selfie preview (see gesture_recognition).
    gesture_mirror_horizontal: bool = True
    # Stabilizer: require sustained agreement before switching motor command.
    gesture_smooth_window: int = 14
    gesture_smooth_min_votes: int = 9
    gesture_smooth_min_votes_none: int = 7
    gesture_stable_min_conf: float = 0.62
    gesture_stable_min_conf_none: float = 0.45

    # --- Motors (gpiozero Robot: left=(fwd, rev), right=(fwd, rev); map pins to M1/M4 on your build) ---
    mock_motors: bool = True
    cruise_speed: float = 0.18
    forward_speed: float = 0.4
    turn_speed: float = 0.35
    reverse_speed: float = 0.35
    u_turn_spin_seconds: float = 2.2
    u_turn_rotate_speed: float = 0.45
    left_motor_pins: Tuple[int, int] = (17, 18)
    right_motor_pins: Tuple[int, int] = (22, 23)
