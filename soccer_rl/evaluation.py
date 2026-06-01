"""Evaluation helpers for comparing simple SoccerRL policies."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

from soccer_rl.envs import Soccer1v1Env
from soccer_rl.envs.soccer_1v1 import DEFAULT_MAX_CYCLES
from soccer_rl.policies import POLICY_NAMES, Policy, make_policy


AGENTS = ("left", "right")
SUMMARY_FIELDNAMES = (
    "episodes",
    "left_policy",
    "right_policy",
    "left_wins",
    "right_wins",
    "truncations",
    "goals",
    "goals_per_episode",
    "avg_episode_length",
    "left_avg_reward",
    "right_avg_reward",
    "left_out_of_bounds",
    "right_out_of_bounds",
    "left_goal_area_violations",
    "right_goal_area_violations",
    "ball_relocations",
)


@dataclass
class EvaluationSummary:
    episodes: int
    left_policy: str
    right_policy: str
    left_wins: int
    right_wins: int
    truncations: int
    avg_episode_length: float
    avg_rewards: dict[str, float]
    out_of_bounds: dict[str, int]
    goal_area_violations: dict[str, int]
    ball_relocations: int

    @property
    def goals(self) -> int:
        return self.left_wins + self.right_wins

    @property
    def goals_per_episode(self) -> float:
        return self.goals / self.episodes


def run_evaluation(
    left_policy: str | Policy = "random",
    right_policy: str | Policy = "random",
    episodes: int = 20,
    seed: int | None = None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
) -> EvaluationSummary:
    """Run headless evaluation episodes and aggregate gameplay metrics."""

    if episodes <= 0:
        raise ValueError("episodes must be greater than 0")

    left = _coerce_policy(left_policy, seed)
    right = _coerce_policy(right_policy, None if seed is None else seed + 1)
    env = Soccer1v1Env(max_cycles=max_cycles)

    wins = {"left": 0, "right": 0}
    total_steps = 0
    total_rewards = {agent: 0.0 for agent in AGENTS}
    out_of_bounds = {agent: 0 for agent in AGENTS}
    goal_area_violations = {agent: 0 for agent in AGENTS}
    truncations = 0
    ball_relocations = 0

    try:
        for episode_index in range(episodes):
            episode_seed = None if seed is None else seed + episode_index
            _reset_policy(left, None if seed is None else seed + 10_000 + episode_index)
            _reset_policy(right, None if seed is None else seed + 20_000 + episode_index)
            observations, _ = env.reset(seed=episode_seed)

            while env.agents:
                actions = {
                    "left": left.act("left", observations["left"]),
                    "right": right.act("right", observations["right"]),
                }
                observations, rewards, terminations, step_truncations, infos = env.step(actions)

                for agent in AGENTS:
                    total_rewards[agent] += rewards[agent]
                    if infos[agent].get("out_of_bounds"):
                        out_of_bounds[agent] += 1
                    if infos[agent].get("goal_area_violation"):
                        goal_area_violations[agent] += 1

                if any(info.get("ball_relocated") for info in infos.values()):
                    ball_relocations += 1

                if all(terminations.values()) or all(step_truncations.values()):
                    break

            total_steps += env.steps
            if env.last_goal is None:
                truncations += 1
            else:
                wins[env.last_goal] += 1

    finally:
        env.close()

    return EvaluationSummary(
        episodes=episodes,
        left_policy=_policy_name(left),
        right_policy=_policy_name(right),
        left_wins=wins["left"],
        right_wins=wins["right"],
        truncations=truncations,
        avg_episode_length=total_steps / episodes,
        avg_rewards={
            agent: total_rewards[agent] / episodes
            for agent in AGENTS
        },
        out_of_bounds=out_of_bounds,
        goal_area_violations=goal_area_violations,
        ball_relocations=ball_relocations,
    )


def format_summary(summary: EvaluationSummary) -> str:
    """Return a compact CLI-friendly evaluation report."""

    return "\n".join(
        [
            "SoccerRL evaluation",
            f"episodes: {summary.episodes}",
            f"policies: left={summary.left_policy}, right={summary.right_policy}",
            (
                "results: "
                f"left wins={summary.left_wins}, "
                f"right wins={summary.right_wins}, "
                f"truncations={summary.truncations}"
            ),
            f"goals per episode: {summary.goals_per_episode:.3f}",
            f"avg episode length: {summary.avg_episode_length:.1f}",
            (
                "avg rewards: "
                f"left={summary.avg_rewards['left']:.3f}, "
                f"right={summary.avg_rewards['right']:.3f}"
            ),
            (
                "out of bounds: "
                f"left={summary.out_of_bounds['left']}, "
                f"right={summary.out_of_bounds['right']}"
            ),
            (
                "goal area violations: "
                f"left={summary.goal_area_violations['left']}, "
                f"right={summary.goal_area_violations['right']}"
            ),
            f"ball relocations: {summary.ball_relocations}",
        ]
    )


def run_matrix(
    policies: Sequence[str] = POLICY_NAMES,
    episodes: int = 20,
    seed: int | None = None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
) -> list[EvaluationSummary]:
    """Run every left/right pairing from the selected policy names."""

    summaries = []
    for matchup_index, (left_policy, right_policy) in enumerate(product(policies, repeat=2)):
        matchup_seed = None if seed is None else seed + matchup_index * 1_000
        summaries.append(
            run_evaluation(
                left_policy=left_policy,
                right_policy=right_policy,
                episodes=episodes,
                seed=matchup_seed,
                max_cycles=max_cycles,
            )
        )
    return summaries


def format_matrix(summaries: Sequence[EvaluationSummary]) -> str:
    """Return a Markdown table for baseline matchup comparisons."""

    lines = [
        "| left | right | L win | R win | trunc | goals/ep | avg len | L rew | R rew | OOB L/R | GA L/R | reloc |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.left_policy} | "
            f"{summary.right_policy} | "
            f"{summary.left_wins} | "
            f"{summary.right_wins} | "
            f"{summary.truncations} | "
            f"{summary.goals_per_episode:.3f} | "
            f"{summary.avg_episode_length:.1f} | "
            f"{summary.avg_rewards['left']:.3f} | "
            f"{summary.avg_rewards['right']:.3f} | "
            f"{summary.out_of_bounds['left']}/{summary.out_of_bounds['right']} | "
            f"{summary.goal_area_violations['left']}/{summary.goal_area_violations['right']} | "
            f"{summary.ball_relocations} |"
        )
    return "\n".join(lines)


def summary_to_record(summary: EvaluationSummary) -> dict[str, int | float | str]:
    """Flatten an evaluation summary for CSV/JSON export."""

    return {
        "episodes": summary.episodes,
        "left_policy": summary.left_policy,
        "right_policy": summary.right_policy,
        "left_wins": summary.left_wins,
        "right_wins": summary.right_wins,
        "truncations": summary.truncations,
        "goals": summary.goals,
        "goals_per_episode": summary.goals_per_episode,
        "avg_episode_length": summary.avg_episode_length,
        "left_avg_reward": summary.avg_rewards["left"],
        "right_avg_reward": summary.avg_rewards["right"],
        "left_out_of_bounds": summary.out_of_bounds["left"],
        "right_out_of_bounds": summary.out_of_bounds["right"],
        "left_goal_area_violations": summary.goal_area_violations["left"],
        "right_goal_area_violations": summary.goal_area_violations["right"],
        "ball_relocations": summary.ball_relocations,
    }


def summaries_to_records(
    summaries: Sequence[EvaluationSummary],
) -> list[dict[str, int | float | str]]:
    """Flatten many summaries for persistent experiment artifacts."""

    return [summary_to_record(summary) for summary in summaries]


def write_summaries_csv(
    summaries: Sequence[EvaluationSummary],
    path: str | Path,
) -> Path:
    """Write evaluation summaries as a flat CSV file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summaries_to_records(summaries))
    return output_path


def write_summaries_json(
    summaries: Sequence[EvaluationSummary],
    path: str | Path,
) -> Path:
    """Write evaluation summaries as JSON records."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summaries_to_records(summaries), output_file, indent=2)
        output_file.write("\n")
    return output_path


def _coerce_policy(policy: str | Policy, seed: int | None) -> Policy:
    if isinstance(policy, str):
        return make_policy(policy, seed=seed)
    _reset_policy(policy, seed)
    return policy


def _reset_policy(policy: Policy, seed: int | None) -> None:
    policy.reset(seed)


def _policy_name(policy: Policy) -> str:
    return getattr(policy, "name", policy.__class__.__name__)
