"""Small scripted policies used for baseline SoccerRL evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from soccer_rl.envs import Action
from soccer_rl.envs.soccer_1v1 import (
    DEFAULT_FIELD_HEIGHT,
    DEFAULT_FIELD_WIDTH,
    KICK_RADIUS,
)


PLAYER_X = 0
PLAYER_Y = 1
BALL_X = 8
BALL_Y = 9
HAS_BALL = 12
OWN_EJECTION_RATIO = 15

POLICY_NAMES = ("random", "chase", "defensive")


class Policy(Protocol):
    """Observation-only policy interface for simple scripted agents."""

    name: str

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state before an episode."""

    def act(self, agent: str, observation: np.ndarray) -> int:
        """Return an integer SoccerRL action."""


@dataclass
class RandomPolicy:
    """Uniform random policy with optional deterministic seeding."""

    seed: int | None = None
    name: str = "random"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, agent: str, observation: np.ndarray) -> int:
        return int(self._rng.integers(0, len(Action)))


@dataclass
class ChaseBallPolicy:
    """Move toward the ball and kick when close enough."""

    name: str = "chase"
    kick_distance: float = KICK_RADIUS * 0.95
    move_deadzone: float = 0.75

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, agent: str, observation: np.ndarray) -> int:
        if observation[OWN_EJECTION_RATIO] > 0.0:
            return int(Action.STAY)

        if observation[HAS_BALL] >= 0.5 or _ball_distance(observation) <= self.kick_distance:
            return int(Action.KICK)

        return int(
            _move_toward(
                observation,
                target_x=float(observation[BALL_X]),
                target_y=float(observation[BALL_Y]),
                deadzone=self.move_deadzone,
            )
        )


@dataclass
class DefensiveChasePolicy:
    """Chase in the defensive half and otherwise guard a home-side lane."""

    name: str = "defensive"
    kick_distance: float = KICK_RADIUS * 0.95
    move_deadzone: float = 0.75
    left_guard_x: float = 0.30
    right_guard_x: float = 0.70

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, agent: str, observation: np.ndarray) -> int:
        if observation[OWN_EJECTION_RATIO] > 0.0:
            return int(Action.STAY)

        if observation[HAS_BALL] >= 0.5 or _ball_distance(observation) <= self.kick_distance:
            return int(Action.KICK)

        ball_x = float(observation[BALL_X])
        ball_y = float(observation[BALL_Y])
        should_chase = ball_x <= 0.5 if agent == "left" else ball_x >= 0.5
        if should_chase:
            target_x = ball_x
        else:
            target_x = self.left_guard_x if agent == "left" else self.right_guard_x

        return int(
            _move_toward(
                observation,
                target_x=target_x,
                target_y=ball_y,
                deadzone=self.move_deadzone,
            )
        )


def make_policy(name: str, seed: int | None = None) -> Policy:
    """Build a scripted policy by CLI-friendly name."""

    normalized = name.lower().replace("-", "_")
    if normalized == "random":
        return RandomPolicy(seed=seed)
    if normalized in {"chase", "chase_ball"}:
        return ChaseBallPolicy()
    if normalized in {"defensive", "defensive_chase"}:
        return DefensiveChasePolicy()

    choices = ", ".join(POLICY_NAMES)
    raise ValueError(f"Unknown policy {name!r}. Choose one of: {choices}.")


def _ball_distance(observation: np.ndarray) -> float:
    dx = (float(observation[BALL_X]) - float(observation[PLAYER_X])) * DEFAULT_FIELD_WIDTH
    dy = (float(observation[BALL_Y]) - float(observation[PLAYER_Y])) * DEFAULT_FIELD_HEIGHT
    return float(np.hypot(dx, dy))


def _move_toward(
    observation: np.ndarray,
    target_x: float,
    target_y: float,
    deadzone: float,
) -> Action:
    dx = (target_x - float(observation[PLAYER_X])) * DEFAULT_FIELD_WIDTH
    dy = (target_y - float(observation[PLAYER_Y])) * DEFAULT_FIELD_HEIGHT
    move_x = abs(dx) > deadzone
    move_y = abs(dy) > deadzone

    if not move_x and not move_y:
        return Action.STAY

    if move_x and (not move_y or abs(dx) >= abs(dy)):
        return Action.RIGHT if dx > 0.0 else Action.LEFT

    return Action.DOWN if dy > 0.0 else Action.UP
