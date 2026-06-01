"""Small scripted policies used for baseline SoccerRL evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from soccer_rl.envs import Action
from soccer_rl.envs.soccer_1v1 import (
    DEFAULT_FIELD_HEIGHT,
    DEFAULT_FIELD_WIDTH,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    KICK_RADIUS,
    PLAYABLE_FIELD_MARGIN,
    PLAYER_RADIUS,
    PLAYER_SPEED,
)


PLAYER_X = 0
PLAYER_Y = 1
BALL_X = 8
BALL_Y = 9
HAS_BALL = 12
OWN_EJECTION_RATIO = 15

POLICY_NAMES = ("random", "chase", "safe_chase", "defensive")

LEFT_GOAL_AREA = (
    PLAYABLE_FIELD_MARGIN / DEFAULT_FIELD_WIDTH,
    (DEFAULT_FIELD_HEIGHT * 0.5 - GOAL_AREA_WIDTH * 0.5) / DEFAULT_FIELD_HEIGHT,
    (PLAYABLE_FIELD_MARGIN + GOAL_AREA_DEPTH) / DEFAULT_FIELD_WIDTH,
    (DEFAULT_FIELD_HEIGHT * 0.5 + GOAL_AREA_WIDTH * 0.5) / DEFAULT_FIELD_HEIGHT,
)
RIGHT_GOAL_AREA = (
    (DEFAULT_FIELD_WIDTH - PLAYABLE_FIELD_MARGIN - GOAL_AREA_DEPTH) / DEFAULT_FIELD_WIDTH,
    LEFT_GOAL_AREA[1],
    (DEFAULT_FIELD_WIDTH - PLAYABLE_FIELD_MARGIN) / DEFAULT_FIELD_WIDTH,
    LEFT_GOAL_AREA[3],
)
GOAL_AREA_TARGET_CLEARANCE = (PLAYER_RADIUS + 0.75) / DEFAULT_FIELD_WIDTH


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
class SafeChaseBallPolicy:
    """Chase the ball while avoiding forbidden goal-area boxes."""

    name: str = "safe_chase"
    kick_distance: float = KICK_RADIUS * 0.95
    move_deadzone: float = 0.75

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, agent: str, observation: np.ndarray) -> int:
        if observation[OWN_EJECTION_RATIO] > 0.0:
            return int(Action.STAY)

        if observation[HAS_BALL] >= 0.5 or _ball_distance(observation) <= self.kick_distance:
            return int(Action.KICK)

        target_x, target_y = _goal_area_safe_target(
            float(observation[BALL_X]),
            float(observation[BALL_Y]),
        )
        action = _move_toward(
            observation,
            target_x=target_x,
            target_y=target_y,
            deadzone=self.move_deadzone,
        )
        return int(_avoid_goal_area_action(observation, action))


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
    if normalized in {"safe_chase", "safe_chase_ball", "goal_area_chase"}:
        return SafeChaseBallPolicy()
    if normalized in {"defensive", "defensive_chase"}:
        return DefensiveChasePolicy()

    choices = ", ".join(POLICY_NAMES)
    raise ValueError(f"Unknown policy {name!r}. Choose one of: {choices}.")


def _ball_distance(observation: np.ndarray) -> float:
    dx = (float(observation[BALL_X]) - float(observation[PLAYER_X])) * DEFAULT_FIELD_WIDTH
    dy = (float(observation[BALL_Y]) - float(observation[PLAYER_Y])) * DEFAULT_FIELD_HEIGHT
    return float(np.hypot(dx, dy))


def _goal_area_safe_target(ball_x: float, ball_y: float) -> tuple[float, float]:
    area = _goal_area_at(ball_x, ball_y)
    if area == "left":
        return min(1.0, LEFT_GOAL_AREA[2] + GOAL_AREA_TARGET_CLEARANCE), ball_y
    if area == "right":
        return max(0.0, RIGHT_GOAL_AREA[0] - GOAL_AREA_TARGET_CLEARANCE), ball_y
    return ball_x, ball_y


def _avoid_goal_area_action(observation: np.ndarray, action: Action) -> Action:
    player_x = float(observation[PLAYER_X])
    player_y = float(observation[PLAYER_Y])
    current_area = _goal_area_at(player_x, player_y)
    if current_area is not None:
        return _escape_goal_area_action(current_area, player_x, player_y)

    projected_x, projected_y = _project_action(player_x, player_y, action)
    projected_area = _goal_area_at(projected_x, projected_y)
    if projected_area is None:
        return action

    return _escape_goal_area_action(projected_area, player_x, player_y)


def _goal_area_at(x: float, y: float) -> str | None:
    if _point_in_area(x, y, LEFT_GOAL_AREA):
        return "left"
    if _point_in_area(x, y, RIGHT_GOAL_AREA):
        return "right"
    return None


def _point_in_area(
    x: float,
    y: float,
    area: tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = area
    return left <= x <= right and top <= y <= bottom


def _escape_goal_area_action(area_name: str, player_x: float, player_y: float) -> Action:
    area = LEFT_GOAL_AREA if area_name == "left" else RIGHT_GOAL_AREA
    left, top, right, bottom = area

    if player_y < top:
        return Action.UP
    if player_y > bottom:
        return Action.DOWN
    if player_x < left:
        return Action.LEFT
    if player_x > right:
        return Action.RIGHT

    return Action.RIGHT if area_name == "left" else Action.LEFT


def _project_action(player_x: float, player_y: float, action: Action) -> tuple[float, float]:
    step_x = PLAYER_SPEED / DEFAULT_FIELD_WIDTH
    step_y = PLAYER_SPEED / DEFAULT_FIELD_HEIGHT
    if action == Action.LEFT:
        return player_x - step_x, player_y
    if action == Action.RIGHT:
        return player_x + step_x, player_y
    if action == Action.UP:
        return player_x, player_y - step_y
    if action == Action.DOWN:
        return player_x, player_y + step_y
    return player_x, player_y


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
