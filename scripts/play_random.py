from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccer_rl.envs import Soccer1v1Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random 1v1 SoccerRL play.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=0, help="0 means run until the window closes.")
    parser.add_argument("--no-render", action="store_true", help="Run without opening a Pygame window.")
    parser.add_argument("--goal-pause", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.no_render else "human"
    env = Soccer1v1Env(render_mode=render_mode)
    observations, _ = env.reset(seed=args.seed)
    episode = 1
    rng_seed = args.seed

    try:
        while args.episodes == 0 or episode <= args.episodes:
            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.possible_agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)

            if render_mode == "human":
                env.render()
                if env.should_close:
                    break
                time.sleep(1.0 / max(args.fps, 1))

            finished = all(terminations.values()) or all(truncations.values())
            if finished:
                if env.last_goal is not None:
                    print(
                        f"Episode {episode}: goal by {env.last_goal} "
                        f"at step {env.last_goal_steps}. Score {env.score['left']}-{env.score['right']}"
                    )
                else:
                    print(f"Episode {episode}: truncated at step {env.steps}.")

                if render_mode == "human" and env.last_goal is not None:
                    end_time = time.time() + max(args.goal_pause, 0.0)
                    while time.time() < end_time and not env.should_close:
                        env.render()
                        time.sleep(1.0 / max(args.fps, 1))

                episode += 1
                if args.episodes != 0 and episode > args.episodes:
                    break

                rng_seed = None if rng_seed is None else rng_seed + 1
                observations, _ = env.reset(seed=rng_seed)

        if args.no_render:
            print(f"Finished {episode - 1} episode(s).")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
