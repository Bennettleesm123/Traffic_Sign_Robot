"""
Motor backends: mock (logging) and Raspberry Pi gpiozero differential drive.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from .sign_policy import RobotAction

log = logging.getLogger(__name__)


class MotorController(ABC):
    @abstractmethod
    def apply(self, action: RobotAction) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def perform_u_turn(self, settings) -> None:
        """Blocking ~180° spin; duration set by ``settings.u_turn_spin_seconds``."""
        ...


class MockMotorController(MotorController):
    """Development / laptop: log actions only."""

    def apply(self, action: RobotAction) -> None:
        log.info("motor: %s", action.name)

    def stop(self) -> None:
        log.info("motor: STOP (coast)")

    def perform_u_turn(self, settings) -> None:
        log.info(
            "motor: U_TURN spin %.2fs at speed %.2f",
            settings.u_turn_spin_seconds,
            settings.u_turn_rotate_speed,
        )
        time.sleep(settings.u_turn_spin_seconds)


class GpioMotorController(MotorController):
    """
    Differential drive via gpiozero.Robot.
    Pins are (forward_pin, backward_pin) per side — typical L298N / DRV8833 wiring.
    """

    def __init__(
        self,
        left_pins: tuple[int, int],
        right_pins: tuple[int, int],
        cruise_speed: float = 0.18,
        forward_speed: float = 0.4,
        turn_speed: float = 0.35,
        reverse_speed: float = 0.35,
        u_turn_rotate_speed: float = 0.45,
    ) -> None:
        from gpiozero import Robot

        self._robot = Robot(left=left_pins, right=right_pins)
        self._cruise_speed = cruise_speed
        self._forward_speed = forward_speed
        self._turn_speed = turn_speed
        self._reverse_speed = reverse_speed
        self._u_turn_rotate_speed = u_turn_rotate_speed

    def stop(self) -> None:
        self._robot.stop()

    def apply(self, action: RobotAction) -> None:
        cs = self._cruise_speed
        s = self._forward_speed
        ts = self._turn_speed
        rs = self._reverse_speed
        if action == RobotAction.STOP:
            self._robot.stop()
        elif action == RobotAction.CRUISE_SLOW:
            self._robot.forward(cs)
        elif action == RobotAction.PROCEED:
            self._robot.forward(s)
        elif action == RobotAction.TURN_LEFT:
            self._robot.left(ts)
        elif action == RobotAction.TURN_RIGHT:
            self._robot.right(ts)
        elif action == RobotAction.REVERSE:
            self._robot.backward(rs)
        elif action == RobotAction.YIELD_PAUSE:
            self._robot.stop()

    def perform_u_turn(self, settings) -> None:
        self._robot.left(self._u_turn_rotate_speed)
        time.sleep(settings.u_turn_spin_seconds)
        self._robot.stop()


def build_motor_controller(settings) -> MotorController:
    if settings.mock_motors:
        return MockMotorController()
    return GpioMotorController(
        left_pins=settings.left_motor_pins,
        right_pins=settings.right_motor_pins,
        cruise_speed=settings.cruise_speed,
        forward_speed=settings.forward_speed,
        turn_speed=settings.turn_speed,
        reverse_speed=settings.reverse_speed,
        u_turn_rotate_speed=settings.u_turn_rotate_speed,
    )
