# SoccerRL

Minimal 1v1 soccer reinforcement-learning sandbox.

This project currently contains the first playable environment slice:

- A PettingZoo-style 1v1 parallel environment
- A Pygame renderer with two random agents moving on a small soccer field
- Scripted baseline policies and an evaluation runner for quick metrics
- Team-colored players, goals, and scoreboard UI
- Goal detection, reward assignment, episode termination, and reset flow
- Ball-contest handling when robots trap the ball between them
- Robot-to-robot collision handling for circular players
- Out-of-bounds ejection when a player crosses the pale inner field line
- Forbidden goal areas drawn as white boxes around each goal
- Smoke tests for reset, step shape, goal rewards, and rendered goal colors

## Setup

Use Python 3.13 for the current pygame wheel support on macOS.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

The local `.venv` is intentionally ignored by Git.

## Run Random Play

```bash
python scripts/play_random.py
```

This opens a Pygame window where two agents choose random actions. When a goal
is scored, the episode resets and the score remains visible. Close the window or
press Escape or Ctrl+C to stop.

For a quick headless smoke test:

```bash
python scripts/play_random.py --no-render --episodes 2 --seed 1
```

Useful script options:

```bash
python scripts/play_random.py --seed 7
python scripts/play_random.py --fps 30
python scripts/play_random.py --episodes 5
```

## Run Manual Play

```bash
python scripts/play_manual.py
```

Controls:

```text
WASD or arrow keys: move the left player
Space: kick
Escape, Ctrl+C, or window close: stop
```

The right player keeps using random actions. This mode is meant for checking how
possession, kicks, wall bounces, and goal timing feel before adding training.

## Evaluate Baseline Agents

```bash
python scripts/evaluate_agents.py --episodes 20 --seed 1
```

Available policies are:

```text
random
chase
defensive
```

Example comparisons:

```bash
python scripts/evaluate_agents.py --left chase --right random --episodes 20 --seed 1
python scripts/evaluate_agents.py --left defensive --right chase --episodes 20 --seed 1
python scripts/compare_baselines.py --episodes 50 --seed 1
```

The report includes wins, truncations, goals per episode, average rewards,
out-of-bounds penalties, goal-area violations, and ball relocations. Use it as a
numeric baseline before changing physics, rewards, or training code.

## Test

```bash
python -m pytest
```

## Milestone Notes

- [v0.0.1 initialization description](update-log-description/v0.0.1-initialization-description.md)
- [v0.0.2 gameplay training foundation](update-log-description/v0.0.2-gameplay-training-foundation-description.md)
- [Current soccer rules](soccer_rules.md)

## Project Structure

```text
SoccerRL/
  soccer_rl/
    evaluation.py        # Baseline evaluation metrics
    policies.py          # Scripted baseline policies
    envs/
      soccer_1v1.py      # 1v1 environment and Pygame renderer
  scripts/
    evaluate_agents.py   # Headless baseline evaluation runner
    compare_baselines.py # Baseline matchup matrix runner
    play_random.py       # Random-agent demo runner
    play_manual.py       # Human-left vs random-right debug runner
  tests/
    test_soccer_env.py   # Environment and render smoke tests
  requirements.txt
```

## Current Environment

`Soccer1v1Env` exposes two agents:

- `left`
- `right`

Actions:

```text
0 stay
1 up
2 down
3 left
4 right
5 kick
```

Observations are compact numeric vectors containing player position, opponent
position, player velocity, opponent velocity, ball position, ball velocity,
possession flags, attack direction, ejection timer state, and ball-contest timer
state.

Rewards are sparse for now:

- `+1.0` for the scoring agent
- `-1.0` for the conceding agent
- `OUT_OF_BOUNDS_PENALTY` when a player crosses the playable inner line
- `GOAL_AREA_PENALTY` when a player enters either goal area box
- `0.0` otherwise

See [soccer_rules.md](soccer_rules.md) for the full current rule set.

## Tuning Physics

Most user-facing physics and rendering values live as module-level constants near
the top of `soccer_rl/envs/soccer_1v1.py`, including:

- field size and goal width
- goal area depth, width, and penalty
- playable field margin and out-of-bounds ejection settings
- player max speed, acceleration, deceleration, and radius
- ball radius, friction, rolling resistance, stop speed, and wall bounce
- possession radius, carried-ball contest radius, and kick radius
- free-ball contest radius and stuck-ball relocation settings
- kick speed and aim limit
- render dimensions and team colors

This is intentionally simple for now: edit the constants, run manual play, then
run tests.

## Notes So Far

The project initially hit a Pygame install issue on Python 3.14 because pip tried
to build Pygame from source and could not find SDL headers. The workspace now
uses Python 3.13.13, where `pygame==2.6.1` installs from a prebuilt wheel on
macOS.

The current development branch is:

```text
RA_v0.0.2-gameplay-training-foundation
```

## Next Likely Steps

- Use `scripts/evaluate_agents.py` to compare random, chase, and defensive
  baselines before tuning rules or rewards
- Continue tuning possession, contests, and kick mechanics toward a RoboCupJunior
  1:1 Infrared/Lightweight-inspired feel
- Add denser reward shaping for learning experiments
- Add a baseline training script after the environment behavior stabilizes
