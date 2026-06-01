import os

from soccer_rl.envs import Action, Soccer1v1Env
from soccer_rl.envs.soccer_1v1 import (
    BALL_FRICTION,
    BALL_RADIUS,
    BALL_ROLLING_RESISTANCE,
    BALL_CONTEST_RADIUS,
    BALL_CONTEST_RELOCATION_STEPS,
    BALL_RELOCATION_MAX_DISTANCE,
    BALL_RELOCATION_MIN_DISTANCE,
    BALL_RELOCATION_PLAYER_CLEARANCE,
    CONTROL_RADIUS,
    DEFAULT_FIELD_HEIGHT,
    DEFAULT_FIELD_WIDTH,
    DEFAULT_GOAL_WIDTH,
    DEFAULT_MAX_CYCLES,
    GOAL_AREA_DEPTH,
    GOAL_AREA_PENALTY,
    GOAL_AREA_WIDTH,
    KICK_RADIUS,
    KICK_SPEED,
    MAX_BALL_SPEED,
    OUT_OF_BOUNDS_EJECTION_STEPS,
    OUT_OF_BOUNDS_PENALTY,
    PLAYABLE_FIELD_MARGIN,
    PLAYER_ACCELERATION,
    PLAYER_DECELERATION,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    TACKLE_RADIUS,
)


def test_default_constants_feed_environment_attributes():
    env = Soccer1v1Env()

    assert env.field_width == DEFAULT_FIELD_WIDTH
    assert env.field_height == DEFAULT_FIELD_HEIGHT
    assert env.goal_width == DEFAULT_GOAL_WIDTH
    assert env.max_cycles == DEFAULT_MAX_CYCLES
    assert env.playable_field_margin == PLAYABLE_FIELD_MARGIN
    assert env.goal_area_depth == GOAL_AREA_DEPTH
    assert env.goal_area_width == GOAL_AREA_WIDTH
    assert env.player_speed == PLAYER_SPEED
    assert env.player_acceleration == PLAYER_ACCELERATION
    assert env.player_deceleration == PLAYER_DECELERATION
    assert env.player_radius == PLAYER_RADIUS
    assert env.ball_radius == BALL_RADIUS
    assert env.control_radius == CONTROL_RADIUS
    assert env.tackle_radius == TACKLE_RADIUS
    assert env.ball_contest_radius == BALL_CONTEST_RADIUS
    assert env.ball_contest_relocation_steps == BALL_CONTEST_RELOCATION_STEPS
    assert env.ball_relocation_min_distance == BALL_RELOCATION_MIN_DISTANCE
    assert env.ball_relocation_max_distance == BALL_RELOCATION_MAX_DISTANCE
    assert env.ball_relocation_player_clearance == BALL_RELOCATION_PLAYER_CLEARANCE
    assert env.kick_radius == KICK_RADIUS
    assert env.kick_speed == KICK_SPEED
    assert env.max_ball_speed == MAX_BALL_SPEED
    assert env.ball_friction == BALL_FRICTION
    assert env.ball_rolling_resistance == BALL_ROLLING_RESISTANCE
    assert env.out_of_bounds_ejection_steps == OUT_OF_BOUNDS_EJECTION_STEPS
    assert env.out_of_bounds_penalty == OUT_OF_BOUNDS_PENALTY
    assert env.goal_area_penalty == GOAL_AREA_PENALTY


def test_reset_returns_two_agents_with_valid_observations():
    env = Soccer1v1Env()

    observations, infos = env.reset(seed=7)

    assert env.agents == ["left", "right"]
    assert set(observations) == {"left", "right"}
    assert set(infos) == {"left", "right"}
    for agent, observation in observations.items():
        assert env.observation_space(agent).contains(observation)


def test_random_step_preserves_parallel_env_shapes():
    env = Soccer1v1Env()
    env.reset(seed=11)
    actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    assert set(observations) == {"left", "right"}
    assert set(rewards) == {"left", "right"}
    assert set(terminations) == {"left", "right"}
    assert set(truncations) == {"left", "right"}
    assert set(infos) == {"left", "right"}


def test_goal_rewards_scorer_and_ends_episode():
    env = Soccer1v1Env()
    env.reset(seed=13)
    env.ball.x = env.field_width + 0.1
    env.ball.y = env.field_height * 0.5

    _, rewards, terminations, truncations, infos = env.step({"left": 0, "right": 0})

    assert rewards["left"] == 1.0
    assert rewards["right"] == -1.0
    assert all(terminations.values())
    assert not any(truncations.values())
    assert infos["left"]["goal_by"] == "left"
    assert env.agents == []


def test_kicked_ball_has_inertia_but_stops_within_short_range():
    env = Soccer1v1Env()
    env.reset(seed=19)
    env.players["left"].y = env.field_height * 0.5
    env.ball.owner = "left"
    env._carry_ball("left")
    start_x = env.ball.x

    env.step({"left": int(Action.KICK), "right": int(Action.STAY)})
    first_speed = (env.ball.vx**2 + env.ball.vy**2) ** 0.5

    for _ in range(80):
        if env.ball.vx == 0.0 and env.ball.vy == 0.0:
            break
        env.step({"left": int(Action.STAY), "right": int(Action.STAY)})

    assert first_speed > 0.0
    assert env.ball.vx == 0.0
    assert env.ball.vy == 0.0
    assert env.ball.x - start_x < env.field_width * 0.35


def test_player_accelerates_to_max_speed_and_coasts_briefly():
    env = Soccer1v1Env()
    env.reset(seed=21)
    start_x = env.players["left"].x

    env.step({"left": int(Action.RIGHT), "right": int(Action.STAY)})

    assert env.players["left"].vx == PLAYER_ACCELERATION
    assert env.players["left"].x == start_x + PLAYER_ACCELERATION

    for _ in range(8):
        env.step({"left": int(Action.RIGHT), "right": int(Action.STAY)})

    assert env.players["left"].vx == PLAYER_SPEED
    before_coast_x = env.players["left"].x

    env.step({"left": int(Action.STAY), "right": int(Action.STAY)})

    coast_distance = env.players["left"].x - before_coast_x
    assert 0.0 < coast_distance <= PLAYER_DECELERATION
    assert coast_distance * 960 / env.field_width < 6.0


def test_player_reverse_input_resets_axis_velocity_before_accelerating():
    env = Soccer1v1Env()
    env.reset(seed=22)

    env.step({"left": int(Action.RIGHT), "right": int(Action.STAY)})
    env.step({"left": int(Action.RIGHT), "right": int(Action.STAY)})
    moving_x = env.players["left"].x

    env.step({"left": int(Action.LEFT), "right": int(Action.STAY)})

    assert env.players["left"].vx == 0.0
    assert env.players["left"].x == moving_x


def test_player_collision_separates_robots_and_stops_closing_velocity():
    env = Soccer1v1Env()
    env.reset(seed=24)
    env.players["left"].x = env.field_width * 0.5
    env.players["left"].y = env.field_height * 0.5
    env.players["left"].vx = PLAYER_SPEED
    env.players["right"].x = env.players["left"].x + env.player_radius
    env.players["right"].y = env.players["left"].y
    env.players["right"].vx = -PLAYER_SPEED

    env._resolve_player_collisions()

    distance = env._distance(env.players["left"], env.players["right"])
    assert distance == env.player_radius * 2.0
    assert env.players["left"].vx == 0.0
    assert env.players["right"].vx == 0.0


def test_nearby_opponent_contests_carried_ball_without_bounce():
    env = Soccer1v1Env()
    env.reset(seed=23)
    env.players["left"].x = env.field_width * 0.45
    env.players["left"].y = env.field_height * 0.5
    env.ball.owner = "left"
    env._carry_ball("left")
    env.players["right"].x = env.ball.x + env.player_radius
    env.players["right"].y = env.ball.y

    env.step({"left": int(Action.STAY), "right": int(Action.STAY)})

    assert env.ball.owner is None
    assert env.ball.vx == 0.0
    assert env.ball.vy == 0.0
    assert env.ball_contest_steps == 1
    assert env.players["left"].x < env.ball.x < env.players["right"].x


def test_contested_ball_relocates_after_timeout():
    env = Soccer1v1Env()
    env.reset(seed=41)
    env.players["left"].x = env.field_width * 0.5 - env.player_radius
    env.players["left"].y = env.field_height * 0.5
    env.players["right"].x = env.field_width * 0.5 + env.player_radius
    env.players["right"].y = env.field_height * 0.5
    env.ball.x = env.field_width * 0.5
    env.ball.y = env.field_height * 0.5
    env.ball.owner = None
    env.ball_contest_steps = BALL_CONTEST_RELOCATION_STEPS - 1

    _, _, _, _, infos = env.step({"left": int(Action.STAY), "right": int(Action.STAY)})

    assert env.ball.owner is None
    assert env.ball_contest_steps == 0
    assert infos["left"]["ball_relocated"] is True
    assert infos["right"]["ball_relocated"] is True
    assert env._is_in_playable_area(env.ball)
    assert not env._is_in_goal_area(env.ball)
    for player in env.players.values():
        assert env._distance(player, env.ball) >= BALL_RELOCATION_PLAYER_CLEARANCE


def test_player_crossing_playable_line_is_ejected_and_penalized():
    env = Soccer1v1Env()
    env.reset(seed=29)
    playable_left, _, _, _ = env._playable_bounds()
    env.players["left"].x = playable_left + env.player_acceleration * 0.5
    env.players["left"].y = env.field_height * 0.5

    _, rewards, terminations, truncations, infos = env.step(
        {"left": int(Action.LEFT), "right": int(Action.STAY)}
    )

    assert rewards["left"] == OUT_OF_BOUNDS_PENALTY
    assert not any(terminations.values())
    assert not any(truncations.values())
    assert infos["left"]["out_of_bounds"] is True
    assert infos["left"]["ejection_steps"] == OUT_OF_BOUNDS_EJECTION_STEPS
    assert env.ejection_timers["left"] == OUT_OF_BOUNDS_EJECTION_STEPS


def test_player_entering_goal_area_is_ejected_and_penalized():
    env = Soccer1v1Env()
    env.reset(seed=30)
    left_goal_area, _ = env._goal_area_bounds()
    _, _, area_right, _ = left_goal_area
    env.players["left"].x = area_right + env.player_acceleration * 0.5
    env.players["left"].y = env.field_height * 0.5

    _, rewards, terminations, truncations, infos = env.step(
        {"left": int(Action.LEFT), "right": int(Action.STAY)}
    )

    assert rewards["left"] == GOAL_AREA_PENALTY
    assert not any(terminations.values())
    assert not any(truncations.values())
    assert infos["left"]["goal_area_violation"] is True
    assert infos["left"]["ejection_steps"] == OUT_OF_BOUNDS_EJECTION_STEPS
    assert env.ejection_timers["left"] == OUT_OF_BOUNDS_EJECTION_STEPS


def test_ejected_player_ignores_actions_and_reenters_after_timeout():
    env = Soccer1v1Env()
    env.reset(seed=31)
    playable_left, _, _, _ = env._playable_bounds()
    env.players["left"].x = playable_left - 0.1
    env.players["left"].y = env.field_height * 0.5
    env.step({"left": int(Action.STAY), "right": int(Action.STAY)})
    ejected_position = (env.players["left"].x, env.players["left"].y)

    env.step({"left": int(Action.RIGHT), "right": int(Action.STAY)})

    assert (env.players["left"].x, env.players["left"].y) == ejected_position

    for _ in range(OUT_OF_BOUNDS_EJECTION_STEPS - 1):
        env.step({"left": int(Action.STAY), "right": int(Action.STAY)})

    assert env.ejection_timers["left"] == 0
    assert env._is_in_playable_area(env.players["left"])
    assert not env._is_in_goal_area(env.players["left"])


def test_render_uses_team_colors_for_goals():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = Soccer1v1Env(render_mode="rgb_array")
    env.reset(seed=17)

    frame = env.render()

    assert frame is not None
    assert tuple(frame[frame.shape[0] // 2, 1]) == env.team_colors["left"]
    assert tuple(frame[frame.shape[0] // 2, frame.shape[1] - 2]) == env.team_colors[
        "right"
    ]


def test_render_draws_goal_area_lines():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = Soccer1v1Env(render_mode="rgb_array")
    env.reset(seed=37)

    frame = env.render()
    left_goal_area, _ = env._goal_area_bounds()
    area_left, _, area_right, _ = left_goal_area
    area_left_px = int(area_left * frame.shape[1] / env.field_width)
    area_width_px = int((area_right - area_left) * frame.shape[1] / env.field_width)
    edge_x = area_left_px + area_width_px
    y = frame.shape[0] // 2

    assert any(
        tuple(frame[y, x]) == (230, 245, 232)
        for x in range(edge_x - 4, edge_x + 1)
    )
