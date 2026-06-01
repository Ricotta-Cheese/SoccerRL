from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccer_rl.envs.soccer_1v1 import DEFAULT_MAX_CYCLES
from soccer_rl.evaluation import format_summary, run_evaluation
from soccer_rl.policies import POLICY_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scripted SoccerRL baseline policies."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=DEFAULT_MAX_CYCLES)
    parser.add_argument("--left", choices=POLICY_NAMES, default="random")
    parser.add_argument("--right", choices=POLICY_NAMES, default="random")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_evaluation(
        left_policy=args.left,
        right_policy=args.right,
        episodes=args.episodes,
        seed=args.seed,
        max_cycles=args.max_cycles,
    )
    print(format_summary(summary))


if __name__ == "__main__":
    main()
