import numpy as np

from soccer_rl.envs import Action
from soccer_rl.evaluation import (
    format_matrix,
    format_summary,
    run_evaluation,
    run_matrix,
    summaries_to_records,
    write_summaries_csv,
    write_summaries_json,
)
from soccer_rl.policies import (
    BALL_X,
    BALL_Y,
    ChaseBallPolicy,
    HAS_BALL,
    LEFT_GOAL_AREA,
    OWN_EJECTION_RATIO,
    PLAYER_X,
    PLAYER_Y,
    RandomPolicy,
    SafeChaseBallPolicy,
)


def test_random_policy_is_seedable_and_returns_valid_actions():
    first = RandomPolicy(seed=4)
    second = RandomPolicy(seed=4)
    observation = np.zeros(18, dtype=np.float32)

    first_actions = [first.act("left", observation) for _ in range(12)]
    second_actions = [second.act("left", observation) for _ in range(12)]

    assert first_actions == second_actions
    assert all(0 <= action < len(Action) for action in first_actions)


def test_chase_policy_moves_toward_ball_and_kicks_when_close():
    policy = ChaseBallPolicy()
    observation = np.zeros(18, dtype=np.float32)
    observation[PLAYER_X] = 0.20
    observation[PLAYER_Y] = 0.50
    observation[BALL_X] = 0.80
    observation[BALL_Y] = 0.50

    assert policy.act("left", observation) == int(Action.RIGHT)

    observation[BALL_X] = observation[PLAYER_X]
    observation[BALL_Y] = observation[PLAYER_Y]

    assert policy.act("left", observation) == int(Action.KICK)

    observation[HAS_BALL] = 0.0
    observation[OWN_EJECTION_RATIO] = 1.0

    assert policy.act("left", observation) == int(Action.STAY)


def test_safe_chase_avoids_stepping_into_goal_area():
    policy = SafeChaseBallPolicy()
    observation = np.zeros(18, dtype=np.float32)
    area_left, area_top, area_right, area_bottom = LEFT_GOAL_AREA
    observation[PLAYER_X] = (area_left + area_right) * 0.5
    observation[PLAYER_Y] = area_top - 0.005
    observation[BALL_X] = (area_left + area_right) * 0.5
    observation[BALL_Y] = (area_top + area_bottom) * 0.5

    assert policy.act("left", observation) == int(Action.UP)


def test_run_evaluation_returns_aggregate_metrics():
    summary = run_evaluation(
        left_policy="random",
        right_policy="random",
        episodes=3,
        seed=9,
        max_cycles=12,
    )

    assert summary.episodes == 3
    assert summary.left_wins + summary.right_wins + summary.truncations == 3
    assert 0 < summary.avg_episode_length <= 12
    assert set(summary.avg_rewards) == {"left", "right"}
    assert set(summary.out_of_bounds) == {"left", "right"}
    assert set(summary.goal_area_violations) == {"left", "right"}
    assert "SoccerRL evaluation" in format_summary(summary)


def test_run_matrix_returns_all_selected_matchups():
    summaries = run_matrix(
        policies=("random", "chase"),
        episodes=1,
        seed=11,
        max_cycles=5,
    )

    assert len(summaries) == 4
    assert {(summary.left_policy, summary.right_policy) for summary in summaries} == {
        ("random", "random"),
        ("random", "chase"),
        ("chase", "random"),
        ("chase", "chase"),
    }
    assert "| left | right |" in format_matrix(summaries)


def test_summary_exports_write_csv_and_json(tmp_path):
    summaries = run_matrix(
        policies=("random", "chase"),
        episodes=1,
        seed=12,
        max_cycles=5,
    )

    csv_path = write_summaries_csv(summaries, tmp_path / "baseline.csv")
    json_path = write_summaries_json(summaries, tmp_path / "baseline.json")
    records = summaries_to_records(summaries)

    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith(
        "episodes,left_policy,right_policy"
    )
    assert '"left_policy": "random"' in json_path.read_text(encoding="utf-8")
    assert records[0]["goals"] == records[0]["left_wins"] + records[0]["right_wins"]
