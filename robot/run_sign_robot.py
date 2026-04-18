"""
**Legacy** traffic-sign + YOLO + HSV traffic-light pipeline (deprecated for the hand-gesture robot).

Use ``python -m robot.run_sign_robot`` if you still need sign detection. The default ``python -m robot``
entry runs ``run_robot`` (MediaPipe gestures).

From the repository root:

  python -m robot.run_sign_robot --weights yolov8n.pt --display
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Sequence

import cv2

from .camera_utils import configure_logitech_style_capture, is_live_camera_source
from .motor_controller import MotorController, build_motor_controller
from .settings import RobotSettings
from .sign_policy import PolicyPipeline, RobotAction, StopSignState, TrafficLightState
from .vision_pipeline import infer_frame
from .yolo_inference import COCO_US_SIGN_HINT, load_yolo_model, model_looks_like_coco80

log = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Legacy CV robot: YOLO signs + traffic lights → motors")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO .pt / .onnx / .engine")
    p.add_argument("--source", default="0", help="Camera index or video path")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--stop-hold", type=float, default=3.0, dest="stop_hold")
    p.add_argument("--yellow-pause", type=float, default=1.5, dest="yellow_pause")
    p.add_argument(
        "--real-motors",
        action="store_true",
        help="Drive gpiozero Robot (default: mock motors / log only)",
    )
    p.add_argument("--left-pins", default="17,18", help="Left motor forward,backward GPIO")
    p.add_argument("--right-pins", default="22,23", help="Right motor forward,backward GPIO")
    p.add_argument("--cruise-speed", type=float, default=0.18)
    p.add_argument("--forward-speed", type=float, default=0.4)
    p.add_argument("--turn-speed", type=float, default=0.35)
    p.add_argument("--uturn-spin", type=float, default=2.2, dest="uturn_spin")
    p.add_argument("--uturn-rotate-speed", type=float, default=0.45, dest="uturn_rotate_speed")
    p.add_argument("--uturn-cooldown", type=float, default=6.0, dest="uturn_cooldown")
    p.add_argument("--display", action="store_true", help="OpenCV window")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-mjpeg", action="store_true")
    p.add_argument(
        "--class-filter",
        choices=("coco_rover", "none"),
        default="coco_rover",
        dest="class_filter",
    )
    p.add_argument("--debug-detect", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def _parse_pin_pair(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return int(a.strip()), int(b.strip())


def run(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    settings = RobotSettings(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        stop_hold_seconds=args.stop_hold,
        yellow_pause_seconds=args.yellow_pause,
        uturn_cooldown_seconds=args.uturn_cooldown,
        mock_motors=not args.real_motors,
        cruise_speed=args.cruise_speed,
        forward_speed=args.forward_speed,
        turn_speed=args.turn_speed,
        u_turn_spin_seconds=args.uturn_spin,
        u_turn_rotate_speed=args.uturn_rotate_speed,
        left_motor_pins=_parse_pin_pair(args.left_pins),
        right_motor_pins=_parse_pin_pair(args.right_pins),
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        camera_prefer_mjpeg=not args.no_mjpeg,
        yolo_class_filter=args.class_filter,
        detection_debug=args.debug_detect,
    )

    log.info("Loading model %s", settings.weights)
    model = load_yolo_model(settings.weights)
    if model_looks_like_coco80(model):
        log.warning(COCO_US_SIGN_HINT)

    policy = PolicyPipeline(
        stop_state=StopSignState(hold_seconds=settings.stop_hold_seconds),
        light_state=TrafficLightState(),
        uturn_cooldown_seconds=settings.uturn_cooldown_seconds,
    )

    motors: MotorController = build_motor_controller(settings)

    src = int(settings.source) if settings.source.isdigit() else settings.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("Could not open video source %s", settings.source)
        sys.exit(1)

    if is_live_camera_source(src):
        configure_logitech_style_capture(
            cap,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
            prefer_mjpeg=settings.camera_prefer_mjpeg,
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            labels, details = infer_frame(model, frame, settings)
            action = policy.step(labels)
            log.debug("labels=%s -> %s", sorted(labels), action.name)

            if action == RobotAction.U_TURN:
                motors.perform_u_turn(settings)
            elif action == RobotAction.YIELD_PAUSE:
                motors.stop()
                time.sleep(settings.yellow_pause_seconds)
            else:
                motors.apply(action)

            if args.display:
                for lab, sc, (x1, y1, x2, y2) in details:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{lab} {sc:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )
                overlay = f"dets={len(details)} labels={sorted(labels)} action={action.name}"
                cv2.putText(
                    frame,
                    overlay,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )
                cv2.imshow("robot_signs_legacy", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
    finally:
        cap.release()
        motors.stop()
        if args.display:
            cv2.destroyAllWindows()


def main() -> None:
    run(None)


if __name__ == "__main__":
    main()
