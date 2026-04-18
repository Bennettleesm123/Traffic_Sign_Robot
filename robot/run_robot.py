"""
Hand-gesture → motors (Raspberry Pi 4B friendly).

Uses **MediaPipe Hands** + geometric rules (no traffic signs / no YOLO on the main path).

From the repository root:

  python -m robot.run_robot --display
  python -m robot.run_robot --real-motors --left-pins 17,18 --right-pins 22,23 --display

Legacy sign detection: ``python -m robot.run_sign_robot``.

**Hardware note:** gpiozero ``Robot`` uses left/right motor pairs. Map GPIO pins to your drivers so they
match **M1 / M4** drive wiring on your chassis (swap left/right pins if the robot turns the wrong way).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

import cv2

from .camera_utils import configure_logitech_style_capture, is_live_camera_source
from .gesture_pipeline import stable_to_action
from .gesture_recognition import GestureRecognizer
from .gesture_smoothing import GestureStabilizer
from .gesture_types import HandGesture
from .motor_controller import MotorController, build_motor_controller
from .settings import RobotSettings
from .sign_policy import RobotAction

log = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand gesture CV robot → gpiozero motors (Pi 4B)")
    p.add_argument("--source", default="0", help="Camera index or video path")
    p.add_argument(
        "--real-motors",
        action="store_true",
        help="Drive gpiozero Robot (default: mock motors / log only)",
    )
    p.add_argument("--left-pins", default="17,18", help="Left motor (forward,backward) GPIO — tie to M1/M4 wiring")
    p.add_argument("--right-pins", default="22,23", help="Right motor (forward,backward) GPIO")
    p.add_argument("--cruise-speed", type=float, default=0.18)
    p.add_argument("--forward-speed", type=float, default=0.4)
    p.add_argument("--turn-speed", type=float, default=0.35)
    p.add_argument("--reverse-speed", type=float, default=0.35)
    p.add_argument("--uturn-spin", type=float, default=2.2, dest="uturn_spin")
    p.add_argument("--uturn-rotate-speed", type=float, default=0.45, dest="uturn_rotate_speed")
    p.add_argument("--display", action="store_true", help="OpenCV window (landmarks + HUD)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-mjpeg", action="store_true")
    # Gesture / MediaPipe
    p.add_argument(
        "--mp-detect-conf",
        type=float,
        default=None,
        help="MediaPipe min hand detection confidence (default: settings)",
    )
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal flip for gesture frame (if left/right feel reversed)",
    )
    p.add_argument("--smooth-window", type=int, default=None, help="Stabilizer rolling window length")
    p.add_argument("--smooth-votes", type=int, default=None, help="Votes required inside window")
    p.add_argument("--min-stab-conf", type=float, default=None, help="Min mean confidence to accept gesture")
    p.add_argument(
        "--debug-gesture",
        action="store_true",
        help="Print each frame: raw gesture + confidence (+ stable output when it changes)",
    )
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

    _d = RobotSettings()
    settings = RobotSettings(
        source=args.source,
        mock_motors=not args.real_motors,
        cruise_speed=args.cruise_speed,
        forward_speed=args.forward_speed,
        turn_speed=args.turn_speed,
        reverse_speed=args.reverse_speed,
        u_turn_spin_seconds=args.uturn_spin,
        u_turn_rotate_speed=args.uturn_rotate_speed,
        left_motor_pins=_parse_pin_pair(args.left_pins),
        right_motor_pins=_parse_pin_pair(args.right_pins),
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        camera_prefer_mjpeg=not args.no_mjpeg,
        gesture_mirror_horizontal=not args.no_mirror,
        gesture_min_detection_conf=args.mp_detect_conf
        if args.mp_detect_conf is not None
        else _d.gesture_min_detection_conf,
        gesture_smooth_window=args.smooth_window
        if args.smooth_window is not None
        else _d.gesture_smooth_window,
        gesture_smooth_min_votes=args.smooth_votes
        if args.smooth_votes is not None
        else _d.gesture_smooth_min_votes,
        gesture_stable_min_conf=args.min_stab_conf
        if args.min_stab_conf is not None
        else _d.gesture_stable_min_conf,
    )

    try:
        recognizer = GestureRecognizer(
            model_complexity=0,
            min_detection_confidence=settings.gesture_min_detection_conf,
            min_tracking_confidence=settings.gesture_min_tracking_conf,
            mirror_horizontal=settings.gesture_mirror_horizontal,
        )
    except ImportError as e:
        log.error("Mediapipe required: pip install mediapipe (%s)", e)
        sys.exit(1)

    stabilizer = GestureStabilizer(
        window_size=settings.gesture_smooth_window,
        min_votes=settings.gesture_smooth_min_votes,
        min_votes_for_none=settings.gesture_smooth_min_votes_none,
        min_mean_conf=settings.gesture_stable_min_conf,
        min_mean_conf_none=settings.gesture_stable_min_conf_none,
    )

    motors: MotorController = build_motor_controller(settings)
    last_action: RobotAction = RobotAction.CRUISE_SLOW
    last_stable_label: Optional[str] = None
    uturn_latched = False  # run blocking U-turn spin once per peace gesture episode

    src = int(settings.source) if settings.source.isdigit() else settings.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("Could not open video source %s", settings.source)
        recognizer.close()
        sys.exit(1)

    if is_live_camera_source(src):
        configure_logitech_style_capture(
            cap,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
            prefer_mjpeg=settings.camera_prefer_mjpeg,
        )

    log.info(
        "Gesture robot: mirror=%s stabilizer window=%s votes≥%s conf≥%.2f (edit robot/gesture_command_map.py)",
        settings.gesture_mirror_horizontal,
        settings.gesture_smooth_window,
        settings.gesture_smooth_min_votes,
        settings.gesture_stable_min_conf,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw_gesture, raw_conf = recognizer.process_frame(frame)
            stable = stabilizer.update(raw_gesture, raw_conf)
            mapped = stable_to_action(stable)

            if mapped is not None:
                last_action = mapped

            if args.debug_gesture:
                sg = f"{stable.gesture.name} mean={stable.mean_confidence:.2f}" if stable else "—"
                line = f"raw={raw_gesture.name} conf={raw_conf:.2f} | stable={sg} | motor={last_action.name}"
                print(line, flush=True)

            if stable is not None:
                lbl = f"{stable.gesture.name}:{stable.mean_confidence:.2f}"
                if lbl != last_stable_label and not args.debug_gesture:
                    log.info("stable gesture %s (votes=%s) -> %s", lbl, stable.votes, last_action.name)
                last_stable_label = lbl

            act = last_action
            if act == RobotAction.U_TURN:
                if not uturn_latched:
                    motors.perform_u_turn(settings)
                    uturn_latched = True
                motors.stop()
            else:
                uturn_latched = False
                motors.apply(act)

            if args.display:
                show = cv2.flip(frame, 1) if settings.gesture_mirror_horizontal else frame
                hud = (
                    f"{raw_gesture.name} raw={raw_conf:.2f} | out={last_action.name} | "
                    f"stab={stable.gesture.name if stable else '...'}"
                )
                cv2.putText(
                    show,
                    hud,
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 240, 255),
                    2,
                )
                cv2.imshow("gesture_robot", show)
                if cv2.waitKey(1) == ord("q"):
                    break
    finally:
        cap.release()
        recognizer.close()
        motors.stop()
        if args.display:
            cv2.destroyAllWindows()


def main() -> None:
    run(None)


if __name__ == "__main__":
    main()
