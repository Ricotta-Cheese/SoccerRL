from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccer_rl.envs import Soccer1v1Env
from soccer_rl.policies import POLICY_NAMES, make_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch scripted SoccerRL baseline policies play in a Pygame window."
    )
    parser.add_argument("--left", choices=POLICY_NAMES, default="safe_chase")
    parser.add_argument("--right", choices=POLICY_NAMES, default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="0 means run until the window closes.",
    )
    parser.add_argument("--goal-pause", type=float, default=0.45)
    return parser.parse_args()


def print_episode_summary(env: Soccer1v1Env, episode: int) -> None:
    if env.last_goal is not None:
        print(
            f"Episode {episode}: goal by {env.last_goal} "
            f"at step {env.last_goal_steps}. Score {env.score['left']}-{env.score['right']}"
        )
    else:
        print(f"Episode {episode}: truncated at step {env.steps}.")


def reset_policies(left_policy, right_policy, seed: int | None, episode: int) -> None:
    if seed is None:
        left_policy.reset()
        right_policy.reset()
        return

    left_policy.reset(seed + 10_000 + episode)
    right_policy.reset(seed + 20_000 + episode)


def main() -> None:
    args = parse_args()
    env = Soccer1v1Env(render_mode="human")
    left_policy = make_policy(args.left, seed=args.seed)
    right_policy = make_policy(
        args.right,
        seed=None if args.seed is None else args.seed + 1,
    )
    observations, _ = env.reset(seed=args.seed)
    env.render()

    print(
        f"Watching left={args.left} vs right={args.right}. "
        "Escape/window close or Ctrl+C to stop."
    )

    episode = 1
    rng_seed = args.seed

    try:
        while args.episodes == 0 or episode <= args.episodes:
            actions = {
                "left": left_policy.act("left", observations["left"]),
                "right": right_policy.act("right", observations["right"]),
            }
            observations, _, terminations, truncations, _ = env.step(actions)

            env.render()
            if env.should_close:
                break

            time.sleep(1.0 / max(args.fps, 1))

            finished = all(terminations.values()) or all(truncations.values())
            if finished:
                print_episode_summary(env, episode)

                if env.last_goal is not None:
                    end_time = time.time() + max(args.goal_pause, 0.0)
                    while time.time() < end_time and not env.should_close:
                        env.render()
                        time.sleep(1.0 / max(args.fps, 1))

                episode += 1
                if args.episodes != 0 and episode > args.episodes:
                    break

                rng_seed = None if rng_seed is None else rng_seed + 1
                reset_policies(left_policy, right_policy, args.seed, episode)
                observations, _ = env.reset(seed=rng_seed)

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
