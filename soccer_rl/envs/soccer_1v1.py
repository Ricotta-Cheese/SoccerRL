from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


DEFAULT_FIELD_WIDTH = 100.0
DEFAULT_FIELD_HEIGHT = 60.0
DEFAULT_GOAL_WIDTH = 22.0
DEFAULT_MAX_CYCLES = 900
RENDER_WIDTH = 960
RENDER_HEIGHT = 576
RENDER_FIELD_MARGIN = 44
PLAYABLE_FIELD_MARGIN = DEFAULT_FIELD_WIDTH * RENDER_FIELD_MARGIN / RENDER_WIDTH
GOAL_AREA_DEPTH = 8.4
GOAL_AREA_WIDTH = DEFAULT_GOAL_WIDTH + 12.0

PLAYER_SPEED = 1.00
PLAYER_ACCELERATION = 0.20
PLAYER_DECELERATION = 0.55
PLAYER_RADIUS = 2.0
BALL_RADIUS = 1.15
BALL_FRICTION = 0.88
BALL_ROLLING_RESISTANCE = 0.064
BALL_STOP_SPEED = 0.06
BALL_WALL_BOUNCE = 0.65
MAX_BALL_SPEED = 4.5

CONTROL_RADIUS = 2.9
TACKLE_RADIUS = 2.9
BALL_CONTEST_RADIUS = 3.2
BALL_CONTEST_RELOCATION_STEPS = 180
BALL_RELOCATION_MIN_DISTANCE = 6.0
BALL_RELOCATION_MAX_DISTANCE = 14.0
BALL_RELOCATION_PLAYER_CLEARANCE = PLAYER_RADIUS + BALL_RADIUS + 1.0
KICK_RADIUS = 4.2
KICK_SPEED = 4.2
KICK_AIM_Y_LIMIT = 0.45
POSSESSION_MAX_SPEED_FACTOR = 1.2
CARRY_BALL_GAP = 0.35
RESET_Y_JITTER = 4.0
OUT_OF_BOUNDS_EJECTION_STEPS = 300
OUT_OF_BOUNDS_PENALTY = -0.20
GOAL_AREA_PENALTY = -0.20

LEFT_START_X_RATIO = 0.24
RIGHT_START_X_RATIO = 0.76

TEAM_COLORS = {
    "left": (229, 72, 77),
    "right": (65, 132, 228),
}


class Action(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    KICK = 5


ACTION_DELTAS = {
    Action.STAY: (0.0, 0.0),
    Action.UP: (0.0, -1.0),
    Action.DOWN: (0.0, 1.0),
    Action.LEFT: (-1.0, 0.0),
    Action.RIGHT: (1.0, 0.0),
    Action.KICK: (0.0, 0.0),
}


@dataclass
class Body:
    x: float
    y: float


@dataclass
class Ball(Body):
    vx: float = 0.0
    vy: float = 0.0
    owner: str | None = None


@dataclass
class Player(Body):
    vx: float = 0.0
    vy: float = 0.0


class Soccer1v1Env(ParallelEnv):
    """A tiny PettingZoo-style 1v1 soccer environment."""

    metadata = {
        "name": "soccer_1v1_v0",
        "render_modes": ["human", "rgb_array"],
    }

    possible_agents = ["left", "right"]
    team_colors = TEAM_COLORS

    def __init__(
        self,
        render_mode: str | None = None,
        field_width: float = DEFAULT_FIELD_WIDTH,
        field_height: float = DEFAULT_FIELD_HEIGHT,
        max_cycles: int = DEFAULT_MAX_CYCLES,
    ) -> None:
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        self.render_mode = render_mode
        self.field_width = field_width
        self.field_height = field_height
        self.goal_width = DEFAULT_GOAL_WIDTH
        self.max_cycles = max_cycles
        self.playable_field_margin = PLAYABLE_FIELD_MARGIN
        self.goal_area_depth = GOAL_AREA_DEPTH
        self.goal_area_width = GOAL_AREA_WIDTH

        self.player_speed = PLAYER_SPEED
        self.player_acceleration = PLAYER_ACCELERATION
        self.player_deceleration = PLAYER_DECELERATION
        self.player_radius = PLAYER_RADIUS
        self.ball_radius = BALL_RADIUS
        self.control_radius = CONTROL_RADIUS
        self.tackle_radius = TACKLE_RADIUS
        self.ball_contest_radius = BALL_CONTEST_RADIUS
        self.ball_contest_relocation_steps = BALL_CONTEST_RELOCATION_STEPS
        self.ball_relocation_min_distance = BALL_RELOCATION_MIN_DISTANCE
        self.ball_relocation_max_distance = BALL_RELOCATION_MAX_DISTANCE
        self.ball_relocation_player_clearance = BALL_RELOCATION_PLAYER_CLEARANCE
        self.kick_radius = KICK_RADIUS
        self.kick_speed = KICK_SPEED
        self.max_ball_speed = MAX_BALL_SPEED
        self.ball_friction = BALL_FRICTION
        self.ball_rolling_resistance = BALL_ROLLING_RESISTANCE
        self.out_of_bounds_ejection_steps = OUT_OF_BOUNDS_EJECTION_STEPS
        self.out_of_bounds_penalty = OUT_OF_BOUNDS_PENALTY
        self.goal_area_penalty = GOAL_AREA_PENALTY

        self.agents: list[str] = []
        self.players: dict[str, Player] = {}
        self.ball = Ball(0.0, 0.0)
        self.ball_contest_steps = 0
        self.ejection_timers = {agent: 0 for agent in self.possible_agents}
        self.score = {"left": 0, "right": 0}
        self.steps = 0
        self.last_goal: str | None = None
        self.last_goal_steps = 0

        self._rng = np.random.default_rng()
        self._screen: Any = None
        self._font: Any = None
        self._small_font: Any = None
        self._closed = False

        self._action_spaces = {
            agent: spaces.Discrete(len(Action)) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent: str) -> spaces.Box:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_spaces[agent]

    @property
    def should_close(self) -> bool:
        return self._closed

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.steps = 0
        self.last_goal = None
        self.last_goal_steps = 0
        self.ball_contest_steps = 0
        self.ejection_timers = {agent: 0 for agent in self.possible_agents}

        jitter_y = float(self._rng.uniform(-RESET_Y_JITTER, RESET_Y_JITTER))
        self.players = {
            "left": Player(
                self.field_width * LEFT_START_X_RATIO,
                self.field_height * 0.5 + jitter_y,
            ),
            "right": Player(
                self.field_width * RIGHT_START_X_RATIO,
                self.field_height * 0.5 - jitter_y,
            ),
        }
        self.ball = Ball(self.field_width * 0.5, self.field_height * 0.5)

        return self._observations(), {agent: {} for agent in self.agents}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        self.steps += 1
        self._tick_ejections()
        clean_actions = {
            agent: int(actions.get(agent, Action.STAY)) for agent in self.possible_agents
        }
        rewards = {agent: 0.0 for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        self._move_players(clean_actions)
        self._resolve_player_collisions()
        self._resolve_player_out_of_bounds(rewards, infos)
        self._resolve_player_goal_area_violations(rewards, infos)
        contest_active = self._resolve_ball_contest(clean_actions, infos)

        if not contest_active:
            kicked_by = self._resolve_kicks(clean_actions)

            if kicked_by is None and self.ball.owner is not None:
                self._carry_ball(self.ball.owner)
            else:
                self._move_free_ball()

            if self.ball.owner is None:
                self._update_possession()

        scorer = self._resolve_goal_or_bounds()
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {
            agent: self.steps >= self.max_cycles for agent in self.possible_agents
        }

        if scorer is not None:
            opponent = self._opponent(scorer)
            rewards[scorer] += 1.0
            rewards[opponent] -= 1.0
            terminations = {agent: True for agent in self.possible_agents}
            truncations = {agent: False for agent in self.possible_agents}
            self.score[scorer] += 1
            self.last_goal = scorer
            self.last_goal_steps = self.steps
            self.ball_contest_steps = 0
            for agent in self.possible_agents:
                infos[agent]["goal_by"] = scorer

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return self._observations(), rewards, terminations, truncations, infos

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        import pygame

        surface = self._render_surface()

        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

        self._ensure_screen()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._closed = True

        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        return None

    def close(self) -> None:
        if self._screen is None:
            return

        import pygame

        pygame.display.quit()
        self._screen = None
        self._font = None
        self._small_font = None

    def _move_players(self, actions: dict[str, int]) -> None:
        for agent, player in self.players.items():
            if self._is_ejected(agent):
                continue

            action = Action(actions[agent])
            dx, dy = ACTION_DELTAS[action]
            player.vx = self._apply_axis_input(player.vx, dx)
            player.vy = self._apply_axis_input(player.vy, dy)

            next_x = player.x + player.vx
            next_y = player.y + player.vy
            player.x = float(np.clip(next_x, 0.0, self.field_width))
            player.y = float(np.clip(next_y, 0.0, self.field_height))

            if player.x != next_x:
                player.vx = 0.0
            if player.y != next_y:
                player.vy = 0.0

    def _apply_axis_input(self, velocity: float, direction: float) -> float:
        if direction == 0.0:
            return self._decelerate_axis(velocity)

        if velocity != 0.0 and np.sign(velocity) != np.sign(direction):
            return 0.0

        velocity += direction * self.player_acceleration
        return float(np.clip(velocity, -self.player_speed, self.player_speed))

    def _decelerate_axis(self, velocity: float) -> float:
        speed = abs(velocity)
        if speed <= self.player_deceleration:
            return 0.0

        return float(np.sign(velocity) * (speed - self.player_deceleration))

    def _resolve_player_collisions(self) -> None:
        first_agent, second_agent = self.possible_agents
        if self._is_ejected(first_agent) or self._is_ejected(second_agent):
            return

        first = self.players[first_agent]
        second = self.players[second_agent]
        dx = second.x - first.x
        dy = second.y - first.y
        distance = float(np.hypot(dx, dy))
        min_distance = self.player_radius * 2.0
        if distance >= min_distance:
            return

        if distance == 0.0:
            nx, ny = 1.0, 0.0
        else:
            nx, ny = dx / distance, dy / distance

        overlap = min_distance - distance
        first.x -= nx * overlap * 0.5
        first.y -= ny * overlap * 0.5
        second.x += nx * overlap * 0.5
        second.y += ny * overlap * 0.5

        self._remove_closing_velocity(first, nx, ny)
        self._remove_closing_velocity(second, -nx, -ny)
        self._clamp_player_to_field(first)
        self._clamp_player_to_field(second)

    def _remove_closing_velocity(self, player: Player, nx: float, ny: float) -> None:
        normal_speed = player.vx * nx + player.vy * ny
        if normal_speed <= 0.0:
            return

        player.vx -= normal_speed * nx
        player.vy -= normal_speed * ny

    def _clamp_player_to_field(self, player: Player) -> None:
        next_x = float(np.clip(player.x, 0.0, self.field_width))
        next_y = float(np.clip(player.y, 0.0, self.field_height))

        if next_x != player.x:
            player.vx = 0.0
        if next_y != player.y:
            player.vy = 0.0

        player.x = next_x
        player.y = next_y

    def _resolve_kicks(self, actions: dict[str, int]) -> str | None:
        if self.ball.owner is not None:
            owner = self.ball.owner
            if self._is_ejected(owner):
                self.ball.owner = None
                return None
            if Action(actions[owner]) == Action.KICK:
                self._kick(owner)
                return owner
            return None

        kickers = [
            agent
            for agent in self.possible_agents
            if not self._is_ejected(agent)
            and Action(actions[agent]) == Action.KICK
            and self._distance(self.players[agent], self.ball) <= self.kick_radius
        ]
        if not kickers:
            return None

        kicker = min(kickers, key=lambda agent: self._distance(self.players[agent], self.ball))
        self._kick(kicker)
        return kicker

    def _kick(self, agent: str) -> None:
        direction = self._attack_direction(agent)
        player = self.players[agent]
        aim_y = self.field_height * 0.5 - player.y
        aim_y = float(
            np.clip(
                aim_y / (self.field_height * 0.5),
                -KICK_AIM_Y_LIMIT,
                KICK_AIM_Y_LIMIT,
            )
        )

        self.ball.owner = None
        self.ball.vx = direction * self.kick_speed
        self.ball.vy = aim_y * self.kick_speed
        self.ball_contest_steps = 0

    def _carry_ball(self, agent: str) -> None:
        player = self.players[agent]
        direction = self._attack_direction(agent)
        self.ball.x = player.x + direction * (
            self.player_radius + self.ball_radius + CARRY_BALL_GAP
        )
        self.ball.y = player.y
        self.ball.vx = 0.0
        self.ball.vy = 0.0

    def _resolve_ball_contest(
        self,
        actions: dict[str, int],
        infos: dict[str, dict[str, Any]],
    ) -> bool:
        if self.ball.owner is not None:
            owner = self.ball.owner
            if Action(actions[owner]) == Action.KICK:
                self.ball_contest_steps = 0
                return False

            self._carry_ball(owner)
            opponent = self._opponent(owner)
            if self._is_ejected(opponent):
                self.ball_contest_steps = 0
                return False

            if self._distance(self.players[opponent], self.ball) <= self.tackle_radius:
                self._pin_contested_ball(infos)
                return True

            self.ball_contest_steps = 0
            return False

        if self._players_contest_free_ball():
            self._pin_contested_ball(infos)
            return True

        self.ball_contest_steps = 0
        return False

    def _players_contest_free_ball(self) -> bool:
        if any(self._is_ejected(agent) for agent in self.possible_agents):
            return False

        first_agent, second_agent = self.possible_agents
        first = self.players[first_agent]
        second = self.players[second_agent]
        if self._distance(first, self.ball) > self.ball_contest_radius:
            return False
        if self._distance(second, self.ball) > self.ball_contest_radius:
            return False

        first_vector = np.array([first.x - self.ball.x, first.y - self.ball.y])
        second_vector = np.array([second.x - self.ball.x, second.y - self.ball.y])
        return float(np.dot(first_vector, second_vector)) <= 0.0

    def _pin_contested_ball(self, infos: dict[str, dict[str, Any]]) -> None:
        first = self.players[self.possible_agents[0]]
        second = self.players[self.possible_agents[1]]
        self.ball.owner = None
        self.ball.vx = 0.0
        self.ball.vy = 0.0
        self.ball.x = (first.x + second.x) * 0.5
        self.ball.y = (first.y + second.y) * 0.5
        self.ball_contest_steps += 1

        if self.ball_contest_steps >= self.ball_contest_relocation_steps:
            self._relocate_contested_ball(infos)

    def _relocate_contested_ball(self, infos: dict[str, dict[str, Any]]) -> None:
        origin_x = self.ball.x
        origin_y = self.ball.y
        for _ in range(48):
            angle = float(self._rng.uniform(0.0, np.pi * 2.0))
            distance = float(
                self._rng.uniform(
                    self.ball_relocation_min_distance,
                    self.ball_relocation_max_distance,
                )
            )
            candidate_x = origin_x + np.cos(angle) * distance
            candidate_y = origin_y + np.sin(angle) * distance
            if self._is_ball_relocation_safe(candidate_x, candidate_y):
                self.ball.x = candidate_x
                self.ball.y = candidate_y
                break
        else:
            left, top, right, bottom = self._playable_bounds()
            self.ball.x = (left + right) * 0.5
            self.ball.y = (top + bottom) * 0.5

        self.ball.owner = None
        self.ball.vx = 0.0
        self.ball.vy = 0.0
        self.ball_contest_steps = 0
        for agent in self.possible_agents:
            infos[agent]["ball_relocated"] = True

    def _is_ball_relocation_safe(self, x: float, y: float) -> bool:
        candidate = Body(x, y)
        if not self._is_in_playable_area(candidate):
            return False
        if self._is_in_goal_area(candidate):
            return False
        return all(
            self._distance(player, candidate) >= self.ball_relocation_player_clearance
            for player in self.players.values()
        )

    def _move_free_ball(self) -> None:
        if self.ball.owner is not None:
            return

        self._clamp_ball_speed()
        self.ball.x += self.ball.vx
        self.ball.y += self.ball.vy
        self._apply_ball_friction()

    def _clamp_ball_speed(self) -> None:
        speed = float(np.hypot(self.ball.vx, self.ball.vy))
        if speed <= self.max_ball_speed:
            return

        scale = self.max_ball_speed / speed
        self.ball.vx *= scale
        self.ball.vy *= scale

    def _apply_ball_friction(self) -> None:
        self.ball.vx *= self.ball_friction
        self.ball.vy *= self.ball_friction

        speed = float(np.hypot(self.ball.vx, self.ball.vy))
        if speed <= BALL_STOP_SPEED:
            self.ball.vx = 0.0
            self.ball.vy = 0.0
            return

        next_speed = max(0.0, speed - self.ball_rolling_resistance)
        if next_speed <= BALL_STOP_SPEED:
            self.ball.vx = 0.0
            self.ball.vy = 0.0
            return

        scale = next_speed / speed
        self.ball.vx *= scale
        self.ball.vy *= scale

    def _update_possession(self) -> None:
        speed = float(np.hypot(self.ball.vx, self.ball.vy))
        if speed > self.player_speed * POSSESSION_MAX_SPEED_FACTOR:
            return

        nearby = [
            agent
            for agent in self.possible_agents
            if not self._is_ejected(agent)
            and self._distance(self.players[agent], self.ball) <= self.control_radius
        ]
        if not nearby:
            return

        owner = min(nearby, key=lambda agent: self._distance(self.players[agent], self.ball))
        self.ball.owner = owner
        self.ball_contest_steps = 0
        self._carry_ball(owner)

    def _resolve_goal_or_bounds(self) -> str | None:
        in_goal_y = abs(self.ball.y - self.field_height * 0.5) <= self.goal_width * 0.5

        if self.ball.x <= 0.0 and in_goal_y:
            return "right"
        if self.ball.x >= self.field_width and in_goal_y:
            return "left"

        if self.ball.y < 0.0:
            self.ball.y = 0.0
            self.ball.vy = abs(self.ball.vy) * BALL_WALL_BOUNCE
        elif self.ball.y > self.field_height:
            self.ball.y = self.field_height
            self.ball.vy = -abs(self.ball.vy) * BALL_WALL_BOUNCE

        if self.ball.x < 0.0:
            self.ball.x = 0.0
            self.ball.vx = abs(self.ball.vx) * BALL_WALL_BOUNCE
        elif self.ball.x > self.field_width:
            self.ball.x = self.field_width
            self.ball.vx = -abs(self.ball.vx) * BALL_WALL_BOUNCE

        return None

    def _observations(self) -> dict[str, np.ndarray]:
        return {agent: self._observation(agent) for agent in self.possible_agents}

    def _observation(self, agent: str) -> np.ndarray:
        opponent = self._opponent(agent)
        player = self.players[agent]
        other = self.players[opponent]

        return np.array(
            [
                player.x / self.field_width,
                player.y / self.field_height,
                np.clip(player.vx / self.player_speed, -1.0, 1.0),
                np.clip(player.vy / self.player_speed, -1.0, 1.0),
                other.x / self.field_width,
                other.y / self.field_height,
                np.clip(other.vx / self.player_speed, -1.0, 1.0),
                np.clip(other.vy / self.player_speed, -1.0, 1.0),
                np.clip(self.ball.x / self.field_width, 0.0, 1.0),
                np.clip(self.ball.y / self.field_height, 0.0, 1.0),
                np.clip(self.ball.vx / self.max_ball_speed, -1.0, 1.0),
                np.clip(self.ball.vy / self.max_ball_speed, -1.0, 1.0),
                1.0 if self.ball.owner == agent else 0.0,
                1.0 if self.ball.owner == opponent else 0.0,
                float(self._attack_direction(agent)),
                self._ejection_ratio(agent),
                self._ejection_ratio(opponent),
                self._ball_contest_ratio(),
            ],
            dtype=np.float32,
        )

    def _resolve_player_out_of_bounds(
        self,
        rewards: dict[str, float],
        infos: dict[str, dict[str, Any]],
    ) -> None:
        for agent in self.possible_agents:
            if self._is_ejected(agent) or self._is_in_playable_area(self.players[agent]):
                continue

            self._eject_player(agent)
            rewards[agent] += self.out_of_bounds_penalty
            infos[agent]["out_of_bounds"] = True
            infos[agent]["ejection_steps"] = self.out_of_bounds_ejection_steps

    def _resolve_player_goal_area_violations(
        self,
        rewards: dict[str, float],
        infos: dict[str, dict[str, Any]],
    ) -> None:
        for agent in self.possible_agents:
            if self._is_ejected(agent) or not self._is_in_goal_area(self.players[agent]):
                continue

            self._eject_player(agent)
            rewards[agent] += self.goal_area_penalty
            infos[agent]["goal_area_violation"] = True
            infos[agent]["ejection_steps"] = self.out_of_bounds_ejection_steps

    def _tick_ejections(self) -> None:
        for agent in self.possible_agents:
            if self.ejection_timers[agent] <= 0:
                continue

            self.ejection_timers[agent] -= 1
            if self.ejection_timers[agent] == 0:
                self.players[agent] = self._reentry_position(agent)

    def _eject_player(self, agent: str) -> None:
        self.ejection_timers[agent] = self.out_of_bounds_ejection_steps
        self.players[agent] = self._reentry_position(agent)
        if self.ball.owner == agent:
            self.ball.owner = None
            self.ball.vx = 0.0
            self.ball.vy = 0.0
            self.ball_contest_steps = 0

    def _reentry_position(self, agent: str) -> Player:
        left_box, right_box = self._goal_area_bounds()
        x = (
            left_box[2] + self.player_radius
            if agent == "left"
            else right_box[0] - self.player_radius
        )
        _, top, _, bottom = self._playable_bounds()
        return Player(x, (top + bottom) * 0.5)

    def _is_in_playable_area(self, body: Body) -> bool:
        left, top, right, bottom = self._playable_bounds()
        return left <= body.x <= right and top <= body.y <= bottom

    def _is_in_goal_area(self, body: Body) -> bool:
        return any(
            left <= body.x <= right and top <= body.y <= bottom
            for left, top, right, bottom in self._goal_area_bounds()
        )

    def _playable_bounds(self) -> tuple[float, float, float, float]:
        margin = self.playable_field_margin
        return (
            margin,
            margin,
            self.field_width - margin,
            self.field_height - margin,
        )

    def _goal_area_bounds(self) -> tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]:
        playable_left, _, playable_right, _ = self._playable_bounds()
        half_height = self.goal_area_width * 0.5
        top = self.field_height * 0.5 - half_height
        bottom = self.field_height * 0.5 + half_height
        left_box = (
            playable_left,
            top,
            playable_left + self.goal_area_depth,
            bottom,
        )
        right_box = (
            playable_right - self.goal_area_depth,
            top,
            playable_right,
            bottom,
        )
        return left_box, right_box

    def _is_ejected(self, agent: str) -> bool:
        return self.ejection_timers[agent] > 0

    def _ejection_ratio(self, agent: str) -> float:
        if self.out_of_bounds_ejection_steps <= 0:
            return 0.0
        return self.ejection_timers[agent] / self.out_of_bounds_ejection_steps

    def _ball_contest_ratio(self) -> float:
        if self.ball_contest_relocation_steps <= 0:
            return 0.0
        return self.ball_contest_steps / self.ball_contest_relocation_steps

    def _render_surface(self) -> Any:
        import pygame

        pygame.init()
        width, height = RENDER_WIDTH, RENDER_HEIGHT
        surface = pygame.Surface((width, height))
        surface.fill((34, 125, 70))

        scale_x = width / self.field_width
        scale_y = height / self.field_height

        def point(body: Body) -> tuple[int, int]:
            return int(body.x * scale_x), int(body.y * scale_y)

        line = (230, 245, 232)
        muted_line = (117, 179, 132)
        for i in range(0, 10):
            stripe = pygame.Rect(i * width // 10, 0, width // 20, height)
            pygame.draw.rect(surface, (38, 134, 75), stripe)

        pygame.draw.rect(surface, line, surface.get_rect(), 4)
        left, top, right, bottom = self._playable_bounds()
        inner_field = pygame.Rect(
            int(left * scale_x),
            int(top * scale_y),
            int((right - left) * scale_x),
            int((bottom - top) * scale_y),
        )
        pygame.draw.rect(surface, muted_line, inner_field, 2)
        for area in self._goal_area_bounds():
            area_left, area_top, area_right, area_bottom = area
            goal_area_rect = pygame.Rect(
                int(area_left * scale_x),
                int(area_top * scale_y),
                int((area_right - area_left) * scale_x),
                int((area_bottom - area_top) * scale_y),
            )
            pygame.draw.rect(surface, line, goal_area_rect, 2)
        pygame.draw.line(surface, line, (width // 2, 0), (width // 2, height), 3)
        pygame.draw.circle(surface, line, (width // 2, height // 2), int(9 * scale_y), 3)

        goal_top = int((self.field_height * 0.5 - self.goal_width * 0.5) * scale_y)
        goal_bottom = int((self.field_height * 0.5 + self.goal_width * 0.5) * scale_y)
        left_goal = pygame.Rect(0, goal_top, 12, goal_bottom - goal_top)
        right_goal = pygame.Rect(width - 12, goal_top, 12, goal_bottom - goal_top)
        self._draw_goal(surface, left_goal, "left")
        self._draw_goal(surface, right_goal, "right")

        for agent, player in self.players.items():
            if self._is_ejected(agent):
                continue

            color = self.team_colors[agent]
            center = point(player)
            radius = int(self.player_radius * scale_y)
            shadow = center[0] + 2, center[1] + 3
            pygame.draw.circle(surface, (25, 60, 42), shadow, radius)
            pygame.draw.circle(surface, color, center, radius)
            pygame.draw.circle(surface, (255, 255, 255), center, radius, 2)
            if self.ball.owner == agent:
                pygame.draw.circle(surface, (255, 236, 143), center, radius + 5, 2)

        ball_center = point(self.ball)
        ball_radius = int(self.ball_radius * scale_y)
        pygame.draw.circle(
            surface,
            (25, 60, 42),
            (ball_center[0] + 2, ball_center[1] + 3),
            ball_radius,
        )
        pygame.draw.circle(surface, (248, 248, 242), ball_center, ball_radius)
        pygame.draw.circle(surface, (30, 30, 30), ball_center, ball_radius, 1)

        self._draw_hud(surface, width)

        return surface

    def _draw_goal(self, surface: Any, rect: Any, agent: str) -> None:
        import pygame

        color = self.team_colors[agent]
        pygame.draw.rect(surface, self._blend(color, (255, 255, 255), 0.2), rect)
        pygame.draw.rect(surface, color, rect, 3)

    def _draw_hud(self, surface: Any, width: int) -> None:
        import pygame

        font = self._get_font()
        small_font = self._get_small_font()

        panel = pygame.Rect(0, 0, 236, 54)
        panel.centerx = width // 2
        panel.y = 12
        pygame.draw.rect(surface, (24, 48, 38), panel, border_radius=8)
        pygame.draw.rect(surface, (255, 255, 255), panel, 2, border_radius=8)

        left_rect = pygame.Rect(panel.left + 8, panel.top + 8, 78, 38)
        right_rect = pygame.Rect(panel.right - 86, panel.top + 8, 78, 38)
        pygame.draw.rect(surface, self.team_colors["left"], left_rect, border_radius=6)
        pygame.draw.rect(surface, self.team_colors["right"], right_rect, border_radius=6)

        left_score = font.render(str(self.score["left"]), True, (255, 255, 255))
        right_score = font.render(str(self.score["right"]), True, (255, 255, 255))
        dash = font.render("-", True, (232, 244, 235))
        surface.blit(left_score, left_score.get_rect(center=left_rect.center))
        surface.blit(right_score, right_score.get_rect(center=right_rect.center))
        surface.blit(dash, dash.get_rect(center=panel.center))

        if self.last_goal is not None:
            color = self.team_colors[self.last_goal]
            label = small_font.render(f"GOAL {self.last_goal.upper()}", True, color)
            label_rect = label.get_rect(center=(width // 2, panel.bottom + 18))
            badge = label_rect.inflate(22, 10)
            pygame.draw.rect(surface, (24, 48, 38), badge, border_radius=6)
            pygame.draw.rect(surface, color, badge, 2, border_radius=6)
            surface.blit(label, label_rect)

    def _blend(
        self,
        color: tuple[int, int, int],
        target: tuple[int, int, int],
        amount: float,
    ) -> tuple[int, int, int]:
        return tuple(
            int(channel + (target[index] - channel) * amount)
            for index, channel in enumerate(color)
        )

    def _ensure_screen(self) -> None:
        import pygame

        if self._screen is None:
            self._screen = pygame.display.set_mode((RENDER_WIDTH, RENDER_HEIGHT))
            pygame.display.set_caption("SoccerRL 1v1 Random Play")

    def _get_font(self) -> Any:
        import pygame

        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("Arial", 28, bold=True)
        return self._font

    def _get_small_font(self) -> Any:
        import pygame

        if self._small_font is None:
            pygame.font.init()
            self._small_font = pygame.font.SysFont("Arial", 16, bold=True)
        return self._small_font

    def _attack_direction(self, agent: str) -> int:
        return 1 if agent == "left" else -1

    def _opponent(self, agent: str) -> str:
        return "right" if agent == "left" else "left"

    def _distance(self, first: Body, second: Body) -> float:
        return float(np.hypot(first.x - second.x, first.y - second.y))
