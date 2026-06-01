from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccer_rl.envs.soccer_1v1 import DEFAULT_MAX_CYCLES
from soccer_rl.evaluation import (
    format_matrix,
    run_matrix,
    write_summaries_csv,
    write_summaries_json,
)
from soccer_rl.policies import POLICY_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare scripted SoccerRL baseline policies as a matchup matrix."
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=DEFAULT_MAX_CYCLES)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--policies",
        choices=POLICY_NAMES,
        nargs="+",
        default=POLICY_NAMES,
        help="Policy names to include on both sides of the matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = run_matrix(
        policies=tuple(args.policies),
        episodes=args.episodes,
        seed=args.seed,
        max_cycles=args.max_cycles,
    )
    print(format_matrix(summaries))
    if args.csv is not None:
        path = write_summaries_csv(summaries, args.csv)
        print(f"\nWrote CSV: {path}")
    if args.json is not None:
        path = write_summaries_json(summaries, args.json)
        print(f"Wrote JSON: {path}")


if __name__ == "__main__":
    main()
