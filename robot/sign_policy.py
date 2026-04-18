"""
Map detector class names → high-level robot commands and timing.

No model is assumed here: wire your YOLO (or other) class names into DEFAULT_LABEL_TO_ACTION.
Traffic lights: many public sign datasets only detect "traffic light" as an object; red/yellow/green
usually needs a second step (crop + color classifier, or a dedicated traffic-light-state model).
For demos, you can use a separate small model or HSV heuristics on the crop (fragile under lighting).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class RobotAction(Enum):
    STOP = auto()
    CRUISE_SLOW = auto()  # default: straight ahead slowly when no relevant signs
    PROCEED = auto()  # explicit go / normal forward (green, straight arrow, etc.)
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    U_TURN = auto()
    REVERSE = auto()
    YIELD_PAUSE = auto()  # short pause (e.g. yellow light)


# Adjust keys to match YOUR exported YOLO dataset class names exactly.
DEFAULT_LABEL_TO_ACTION: Dict[str, RobotAction] = {
    # Stop / regulatory
    "stop": RobotAction.STOP,
    "stop sign": RobotAction.STOP,
    "stop_sign": RobotAction.STOP,
    # U-turn (MUTCD R3-19a; match YOLO ``names`` from training)
    "u turn": RobotAction.U_TURN,
    "u-turn": RobotAction.U_TURN,
    "uturn": RobotAction.U_TURN,
    "u turn only": RobotAction.U_TURN,
    "u-turn only": RobotAction.U_TURN,
    "u_turn_only": RobotAction.U_TURN,
    "lane for u turn only": RobotAction.U_TURN,
    # Turn-only lanes (PDF / MUTCD R3-5 style)
    "left turn only": RobotAction.TURN_LEFT,
    "right turn only": RobotAction.TURN_RIGHT,
    "left_turn_only": RobotAction.TURN_LEFT,
    "right_turn_only": RobotAction.TURN_RIGHT,
    # Arrows / turn (examples — rename to match your weights)
    "turn left": RobotAction.TURN_LEFT,
    "left": RobotAction.TURN_LEFT,
    "arrow left": RobotAction.TURN_LEFT,
    "turn right": RobotAction.TURN_RIGHT,
    "right": RobotAction.TURN_RIGHT,
    "arrow right": RobotAction.TURN_RIGHT,
    "straight": RobotAction.PROCEED,
    "ahead only": RobotAction.PROCEED,
    "forward": RobotAction.PROCEED,
    # Traffic light *states* if your model has them (uncommon in single-stage sign sets)
    "red light": RobotAction.STOP,
    "yellow light": RobotAction.YIELD_PAUSE,
    "green light": RobotAction.PROCEED,
    # Generic box before HSV, or HSV failed — stay conservative
    "traffic light": RobotAction.STOP,
}


def map_detection_to_action(
    label: str,
    table: Optional[Dict[str, RobotAction]] = None,
) -> Optional[RobotAction]:
    t = table or DEFAULT_LABEL_TO_ACTION
    key = (label or "").strip().lower()
    if key in t:
        return t[key]
    for k, act in t.items():
        if k in key or key in k:
            return act
    return None


def _traffic_light_actions_from_labels(
    labels: Set[str],
    table: Dict[str, RobotAction],
) -> List[RobotAction]:
    """Treat labels that contain 'light' and map to stop/yield/proceed as traffic-light states."""
    out: List[RobotAction] = []
    for lab in labels:
        if "light" not in lab.lower():
            continue
        a = map_detection_to_action(lab, table)
        if a in (RobotAction.STOP, RobotAction.YIELD_PAUSE, RobotAction.PROCEED):
            out.append(a)
    return out


@dataclass
class StopSignState:
    """After a *stop sign* STOP, wait `hold_seconds` then proceed."""

    hold_seconds: float = 3.0
    _stop_seen_at: Optional[float] = None

    def observe(self, action: Optional[RobotAction], now: Optional[float] = None) -> RobotAction:
        t = now if now is not None else time.monotonic()
        if action == RobotAction.STOP:
            self._stop_seen_at = t
            return RobotAction.STOP
        if self._stop_seen_at is not None:
            if t - self._stop_seen_at < self.hold_seconds:
                return RobotAction.STOP
            self._stop_seen_at = None
        if action is not None:
            return action
        return RobotAction.CRUISE_SLOW


@dataclass
class TrafficLightState:
    """
    Red: hold until green is seen. Yellow: brief yield. Green: go.
    If no light is visible but we are waiting after red, stay stopped.
    """

    waiting_for_green: bool = False

    def observe(self, light_actions: List[RobotAction]) -> RobotAction:
        if not light_actions:
            if self.waiting_for_green:
                return RobotAction.STOP
            return RobotAction.CRUISE_SLOW

        if RobotAction.PROCEED in light_actions:
            self.waiting_for_green = False
            return RobotAction.PROCEED
        if RobotAction.YIELD_PAUSE in light_actions:
            return RobotAction.YIELD_PAUSE
        if RobotAction.STOP in light_actions:
            self.waiting_for_green = True
            return RobotAction.STOP
        return RobotAction.STOP


@dataclass
class PolicyPipeline:
    """Traffic-light logic overrides stop-sign timer when lights are in use."""

    stop_state: StopSignState = field(default_factory=StopSignState)
    light_state: TrafficLightState = field(default_factory=TrafficLightState)
    label_table: Dict[str, RobotAction] = field(default_factory=lambda: dict(DEFAULT_LABEL_TO_ACTION))
    uturn_cooldown_seconds: float = 6.0
    _uturn_allowed_at: float = 0.0

    def step(self, detected_labels: Set[str]) -> RobotAction:
        """
        Priority when multiple objects appear:
        traffic-light state > stop > yield > u-turn > left/right > other > empty → CRUISE_SLOW.
        """
        now = time.monotonic()
        lt = self.label_table
        light_acts = _traffic_light_actions_from_labels(detected_labels, lt)
        if light_acts or self.light_state.waiting_for_green:
            return self.light_state.observe(light_acts)

        actions = [map_detection_to_action(x, lt) for x in detected_labels]
        actions = [a for a in actions if a is not None]
        if now < self._uturn_allowed_at:
            actions = [a for a in actions if a != RobotAction.U_TURN]

        raw: Optional[RobotAction]
        if RobotAction.STOP in actions:
            raw = RobotAction.STOP
        elif RobotAction.YIELD_PAUSE in actions:
            raw = RobotAction.YIELD_PAUSE
        elif RobotAction.U_TURN in actions:
            raw = RobotAction.U_TURN
        elif RobotAction.TURN_LEFT in actions or RobotAction.TURN_RIGHT in actions:
            raw = RobotAction.TURN_LEFT if RobotAction.TURN_LEFT in actions else RobotAction.TURN_RIGHT
        elif actions:
            raw = actions[0]
        else:
            raw = RobotAction.CRUISE_SLOW

        final = self.stop_state.observe(raw, None)
        if final == RobotAction.U_TURN:
            self._uturn_allowed_at = now + self.uturn_cooldown_seconds
        return final
