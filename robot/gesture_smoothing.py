"""
Temporal smoothing: require agreement across many frames before accepting a gesture.

Reduces jitter from noisy landmark detection (lighting, motion blur on Raspberry Pi camera).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from .gesture_types import HandGesture


@dataclass
class StableReading:
    gesture: HandGesture
    mean_confidence: float
    votes: int


class GestureStabilizer:
    """
    Rolling window + majority vote + minimum mean confidence.

    Emits a stable ``(gesture, confidence)`` only when:
    - the same non-NONE gesture appears at least ``min_votes`` times in the last ``window_size`` frames, OR
    - ``NONE`` / unrecognized streak dominates (so the robot can fall back to idle cruise).

    ``NONE`` uses a separate lower threshold so we don't stay stuck on an old command when the hand leaves the frame.
    """

    def __init__(
        self,
        window_size: int = 14,
        min_votes: int = 9,
        min_votes_for_none: int = 7,
        min_mean_conf: float = 0.62,
        min_mean_conf_none: float = 0.45,
    ) -> None:
        self._window_size = max(3, window_size)
        self._min_votes = max(1, min_votes)
        self._min_votes_for_none = max(1, min_votes_for_none)
        self._min_mean_conf = min_mean_conf
        self._min_mean_conf_none = min_mean_conf_none
        self._buf: Deque[Tuple[HandGesture, float]] = deque(maxlen=self._window_size)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, gesture: HandGesture, confidence: float) -> Optional[StableReading]:
        """Append one frame; return a stable reading, or None to keep prior motor command."""
        self._buf.append((gesture, confidence))
        if len(self._buf) < self._window_size // 2:
            return None

        gests: List[HandGesture] = [g for g, _ in self._buf]
        confs: List[float] = [c for _, c in self._buf]
        ctr = Counter(gests)
        winner, vcount = ctr.most_common(1)[0]

        mean_c = sum(c for g, c in self._buf if g == winner) / max(1, vcount)

        if winner == HandGesture.NONE:
            need = self._min_votes_for_none
            thr = self._min_mean_conf_none
        else:
            need = self._min_votes
            thr = self._min_mean_conf

        if vcount < need or mean_c < thr:
            return None

        return StableReading(gesture=winner, mean_confidence=mean_c, votes=vcount)
