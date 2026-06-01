import os

from soccer_rl.envs import Action, Soccer1v1Env
from soccer_rl.envs.soccer_1v1 import (
    BALL_FRICTION,
    BALL_RADIUS,
    BALL_ROLLING_RESISTANCE,
    CONTROL_RADIUS,
    DEFAULT_FIELD_HEIGHT,
    DEFAULT_FIELD_WIDTH,
    DEFAULT_GOAL_WIDTH,
    DEFAULT_MAX_CYCLES,
    KICK_RADIUS,
    KICK_SPEED,
    MAX_BALL_SPEED,
    PLAYER_RADIUS,
    PLAYER_SPEED,
)


def test_default_constants_feed_environment_attributes():
    env = Soccer1v1Env()

    assert env.field_width == DEFAULT_FIELD_WIDTH
    assert env.field_height == DEFAULT_FIELD_HEIGHT
    assert env.goal_width == DEFAULT_GOAL_WIDTH
    assert env.max_cycles == DEFAULT_MAX_CYCLES
    assert env.player_speed == PLAYER_SPEED
    assert env.player_radius == PLAYER_RADIUS
    assert env.ball_radius == BALL_RADIUS
    assert env.control_radius == CONTROL_RADIUS
    assert env.kick_radius == KICK_RADIUS
    assert env.kick_speed == KICK_SPEED
    assert env.max_ball_speed == MAX_BALL_SPEED
    assert env.ball_friction == BALL_FRICTION
    assert env.ball_rolling_resistance == BALL_ROLLING_RESISTANCE


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
