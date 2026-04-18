"""
Laptop / Pi desktop: **preview + numeric debug** for gesture recognition (no motors by default).

  python -m robot.gesture_camera_test
  python -m robot.gesture_camera_test --debug-gesture

Press **q** in the OpenCV window to quit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

import cv2

from .camera_utils import configure_logitech_style_capture, is_live_camera_source
from .gesture_pipeline import stable_to_action
from .gesture_recognition import GestureRecognizer
from .gesture_smoothing import GestureStabilizer
from .settings import RobotSettings

log = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test MediaPipe hand gestures + stabilizer (no GPIO)")
    p.add_argument("--source", default="0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-mjpeg", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--debug-gesture", action="store_true", help="Print raw + stable lines to stdout")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    d = RobotSettings()
    try:
        rec = GestureRecognizer(
            model_complexity=0,
            min_detection_confidence=d.gesture_min_detection_conf,
            min_tracking_confidence=d.gesture_min_tracking_conf,
            mirror_horizontal=not args.no_mirror,
        )
    except ImportError as e:
        log.error("Install mediapipe: pip install mediapipe (%s)", e)
        sys.exit(1)

    stab = GestureStabilizer(
        window_size=d.gesture_smooth_window,
        min_votes=d.gesture_smooth_min_votes,
        min_votes_for_none=d.gesture_smooth_min_votes_none,
        min_mean_conf=d.gesture_stable_min_conf,
        min_mean_conf_none=d.gesture_stable_min_conf_none,
    )

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("Could not open %s", args.source)
        rec.close()
        sys.exit(1)
    if is_live_camera_source(src):
        configure_logitech_style_capture(
            cap,
            width=args.width,
            height=args.height,
            fps=args.fps,
            prefer_mjpeg=not args.no_mjpeg,
        )

    win = "gesture_camera_test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_g, raw_c = rec.process_frame(frame)
            st = stab.update(raw_g, raw_c)
            mapped = stable_to_action(st)

            if args.debug_gesture:
                sg = f"{st.gesture.name} mean={st.mean_confidence:.2f}" if st else "—"
                print(f"raw={raw_g.name} conf={raw_c:.2f} | stable={sg} | action={mapped}", flush=True)

            disp = cv2.flip(frame, 1) if not args.no_mirror else frame
            line = f"{raw_g.name} ({raw_c:.2f})  ->  {mapped}  [{st.gesture.name if st else '...'}]"
            cv2.putText(disp, line, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        cap.release()
        rec.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
