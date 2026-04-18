"""
Second-stage traffic light state from a cropped BGR image (HSV heuristics).

Fragile under strong backlight or color cast; good enough for controlled indoor demos.
Returns labels compatible with sign_policy: \"red light\", \"yellow light\", \"green light\".
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def classify_traffic_light_hsv(
    crop_bgr: np.ndarray,
    min_fill_ratio: float = 0.02,
) -> Optional[str]:
    """
    Pick dominant lit signal among red / yellow / green using HSV masks.
    Returns None if the crop is too small or ambiguous.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return None

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # Ignore very dark pixels (background / housing)
    v = hsv[:, :, 2]
    lit = v > 40
    lit_count = max(int(np.count_nonzero(lit)), 1)

    lit_u8 = lit.astype(np.uint8) * 255

    def masked_count(lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> int:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        rng = cv2.inRange(hsv, lo, hi)
        return int(np.count_nonzero(cv2.bitwise_and(rng, lit_u8)))

    # Red wraps hue; yellow and green single ranges (OpenCV H: 0–180)
    red_a = masked_count((0, 60, 40), (12, 255, 255))
    red_b = masked_count((168, 60, 40), (180, 255, 255))
    red = red_a + red_b

    yellow = masked_count((15, 60, 40), (38, 255, 255))
    green = masked_count((38, 50, 40), (95, 255, 255))

    scores = {"red light": red, "yellow light": yellow, "green light": green}
    best_label = max(scores, key=scores.get)
    best = scores[best_label]

    if best / lit_count < min_fill_ratio:
        return None

    # Require clear winner (avoid ties on noise)
    second = sorted(scores.values(), reverse=True)[1]
    if second * 1.15 >= best:
        return None

    return best_label
