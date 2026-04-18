"""
USB webcam capture tuning (e.g. Logitech 720p on Raspberry Pi + OpenCV).

Many Logitech cams need MJPEG (`CAP_PROP_FOURCC`) to reach 1280x720 at usable FPS over USB.
"""

from __future__ import annotations

import logging
from typing import Union

import cv2

log = logging.getLogger(__name__)


def is_live_camera_source(source: Union[str, int]) -> bool:
    if isinstance(source, int):
        return True
    s = str(source).strip()
    if s.isdigit():
        return True
    return s.startswith("/dev/video")


def configure_logitech_style_capture(
    cap: cv2.VideoCapture,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    *,
    prefer_mjpeg: bool = True,
) -> None:
    """
    Request 720p (or other) resolution. ``prefer_mjpeg`` helps most USB HD webcams on Linux/Pi.
    """
    if prefer_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    aw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ah = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    af = cap.get(cv2.CAP_PROP_FPS)
    log.info(
        "Camera: requested %dx%d @ %.1f fps (MJPEG=%s); actual %.0fx%.0f @ %.1f fps",
        width,
        height,
        fps,
        prefer_mjpeg,
        aw,
        ah,
        af,
    )
