"""
Hand pose → ``HandGesture`` using **MediaPipe Tasks Hand Landmarker** + geometric rules.

Uses the official ``hand_landmarker.task`` (downloaded once into a local cache). This matches
MediaPipe 0.10+ which removed ``mp.solutions.hands`` — Pi 4B / modern Python need the Tasks API.

**HaGRID** is a large static-gesture image dataset for CNN training; we cite it as optional future work
if you swap this module for a TFLite classifier. For real-time Pi control, landmark rules stay lighter.
"""

from __future__ import annotations

import logging
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .gesture_types import HandGesture

log = logging.getLogger(__name__)

# Google AI Edge bundle (float16 TFLite) — same family Pi / desktop Tasks API expects.
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _default_model_path() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "robot_gesture_cv"
    root.mkdir(parents=True, exist_ok=True)
    return root / "hand_landmarker.task"


def ensure_hand_landmarker_model(path: Optional[Path] = None) -> Path:
    """Return path to ``.task`` file, downloading on first use if missing."""
    p = path or _default_model_path()
    if p.is_file():
        return p
    log.info("Downloading MediaPipe hand_landmarker model to %s", p)
    urllib.request.urlretrieve(_HAND_MODEL_URL, p)  # nosec B310 — fixed Google URL
    return p


def _dist(a, b) -> float:
    ax = a.x if a.x is not None else 0.0
    ay = a.y if a.y is not None else 0.0
    bx = b.x if b.x is not None else 0.0
    by = b.y if b.y is not None else 0.0
    return math.hypot(ax - bx, ay - by)


class _LM:
    """Tiny wrapper so rules can use ``lm[i].x`` like classic MediaPipe."""

    __slots__ = ("_seq",)

    def __init__(self, seq: Sequence[object]) -> None:
        self._seq = seq

    def __getitem__(self, i: int):
        return self._seq[i]


class GestureRecognizer:
    """
    One BGR frame in → ``(HandGesture, confidence)``.

    ``model_complexity`` is kept for API compatibility; Tasks API uses a single bundled landmarker.
    """

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.5,
        mirror_horizontal: bool = True,
        model_path: Optional[Path] = None,
    ) -> None:
        del model_complexity  # Tasks HandLandmarker has a single model asset
        self._mirror = mirror_horizontal
        mp_image_mod = __import__(
            "mediapipe.tasks.python.vision.core.image",
            fromlist=["Image", "ImageFormat"],
        )
        self._Image = mp_image_mod.Image
        self._ImageFormat = mp_image_mod.ImageFormat

        from mediapipe.tasks.python.core import base_options as base_opts
        from mediapipe.tasks.python.vision import hand_landmarker as hl
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as vmode

        asset = ensure_hand_landmarker_model(model_path)
        opts = hl.HandLandmarkerOptions(
            base_options=base_opts.BaseOptions(model_asset_path=str(asset)),
            running_mode=vmode.VisionTaskRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = hl.HandLandmarker.create_from_options(opts)
        self._t0_ns = time.monotonic_ns()

    def close(self) -> None:
        self._landmarker.close()

    def classify_landmarks(self, lm: _LM) -> Tuple[HandGesture, float]:
        """Rule-based pose classification on one hand's 21 normalized landmarks."""

        vis = [lm[i].visibility for i in (0, 5, 8, 12)]
        base_vis = float(min((v if v is not None else 0.5) for v in vis))
        if base_vis < 0.35:
            return HandGesture.NONE, 0.25

        w = lm[0]
        # index / middle / ring / pinky extended: tip farther from wrist than PIP
        def ext4(tip: int, pip: int) -> bool:
            vv = lm[tip].visibility
            if vv is not None and vv < 0.45:
                return False
            return _dist(lm[tip], w) > _dist(lm[pip], w) * 1.12

        idx = ext4(8, 6)
        mid = ext4(12, 10)
        ring = ext4(16, 14)
        pink = ext4(20, 18)

        t_tip, t_ip, t_mcp = lm[4], lm[3], lm[2]
        thumb_long = _dist(t_tip, w) > _dist(t_ip, w) * 1.12

        thumb_points_up = thumb_long and t_tip.y is not None and t_ip.y is not None and t_mcp.y is not None and (t_tip.y < t_ip.y < t_mcp.y + 0.02)  # noqa: E501
        thumb_points_down = thumb_long and t_tip.y is not None and t_ip.y is not None and t_mcp.y is not None and (t_tip.y > t_ip.y > t_mcp.y - 0.02)  # noqa: E501

        four = (idx, mid, ring, pink)
        n_ext = sum(four)

        if idx and mid and (not ring) and (not pink):
            vv = min((lm[i].visibility or 0.8) for i in (8, 12, 0))
            return HandGesture.PEACE, float(min(1.0, 0.5 + 0.4 * vv))

        if idx and n_ext == 1:
            tip_x = lm[8].x
            wx = w.x
            if tip_x is None or wx is None:
                return HandGesture.NONE, 0.4
            edge = 0.022
            vis_c = min((lm[8].visibility or 0.8), (w.visibility or 0.8))
            if tip_x < wx - edge:
                return HandGesture.POINT_LEFT, float(min(1.0, 0.5 + 0.45 * vis_c))
            if tip_x > wx + edge:
                return HandGesture.POINT_RIGHT, float(min(1.0, 0.5 + 0.45 * vis_c))
            return (
                HandGesture.POINT_LEFT if tip_x < wx else HandGesture.POINT_RIGHT,
                float(0.45 + 0.35 * vis_c),
            )

        if not any((idx, mid, ring, pink)):
            tv = lm[4].visibility or 0.8
            if thumb_points_up:
                return HandGesture.THUMBS_UP, float(min(1.0, 0.55 + 0.35 * tv))
            if thumb_points_down:
                return HandGesture.THUMBS_DOWN, float(min(1.0, 0.5 + 0.35 * tv))

        if n_ext == 4:
            vv = min((lm[i].visibility or 0.8) for i in (8, 12, 16, 20))
            return HandGesture.OPEN_PALM, float(min(1.0, 0.48 + 0.45 * vv))

        if n_ext == 0 and not thumb_long:
            return HandGesture.FIST, float(min(1.0, 0.45 + 0.4 * base_vis))

        return HandGesture.NONE, float(0.35 + 0.25 * base_vis)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[HandGesture, float]:
        """BGR frame from OpenCV → optional flip → RGB → HandLandmarker (VIDEO)."""
        inf = cv2.flip(frame_bgr, 1) if self._mirror else frame_bgr
        rgb = cv2.cvtColor(inf, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_img = self._Image(self._ImageFormat.SRGB, rgb)

        ts_ms = (time.monotonic_ns() - self._t0_ns) // 1_000_000
        res = self._landmarker.detect_for_video(mp_img, int(ts_ms))

        if not res.hand_landmarks:
            return HandGesture.NONE, 0.2

        lm_list: List = list(res.hand_landmarks[0])
        return self.classify_landmarks(_LM(lm_list))
