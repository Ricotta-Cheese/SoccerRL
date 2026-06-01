from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccer_rl.envs import Action, Soccer1v1Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control the left player against a random right player."
    )
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


def action_from_keys(keys, pygame_module) -> Action:
    if keys[pygame_module.K_SPACE]:
        return Action.KICK
    if keys[pygame_module.K_w] or keys[pygame_module.K_UP]:
        return Action.UP
    if keys[pygame_module.K_s] or keys[pygame_module.K_DOWN]:
        return Action.DOWN
    if keys[pygame_module.K_a] or keys[pygame_module.K_LEFT]:
        return Action.LEFT
    if keys[pygame_module.K_d] or keys[pygame_module.K_RIGHT]:
        return Action.RIGHT
    return Action.STAY


def print_episode_summary(env: Soccer1v1Env, episode: int) -> None:
    if env.last_goal is not None:
        print(
            f"Episode {episode}: goal by {env.last_goal} "
            f"at step {env.last_goal_steps}. Score {env.score['left']}-{env.score['right']}"
        )
    else:
        print(f"Episode {episode}: truncated at step {env.steps}.")


def main() -> None:
    import pygame

    args = parse_args()
    env = Soccer1v1Env(render_mode="human")
    env.reset(seed=args.seed)
    env.render()

    print(
        "Controls: WASD or arrow keys to move, Space to kick, "
        "Escape/window close to stop."
    )

    clock = pygame.time.Clock()
    episode = 1
    rng_seed = args.seed

    try:
        while args.episodes == 0 or episode <= args.episodes:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            actions = {
                "left": int(action_from_keys(keys, pygame)),
                "right": env.action_space("right").sample(),
            }
            _, _, terminations, truncations, _ = env.step(actions)

            env.render()
            if env.should_close:
                break

            finished = all(terminations.values()) or all(truncations.values())
            if finished:
                print_episode_summary(env, episode)

                if env.last_goal is not None:
                    end_time = time.time() + max(args.goal_pause, 0.0)
                    while time.time() < end_time and not env.should_close:
                        env.render()
                        clock.tick(max(args.fps, 1))

                episode += 1
                if args.episodes != 0 and episode > args.episodes:
                    break

                rng_seed = None if rng_seed is None else rng_seed + 1
                env.reset(seed=rng_seed)

            clock.tick(max(args.fps, 1))
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
