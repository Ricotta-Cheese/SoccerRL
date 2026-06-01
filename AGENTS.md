# AGENTS.md

Guidance for coding agents working in this repository.

## Project Snapshot

SoccerRL is currently a small Python reinforcement-learning sandbox centered on
a minimal 1v1 soccer environment.

The first milestone is a playable/debuggable random-agent demo:

```bash
python scripts/play_random.py
```

The demo opens a Pygame window, moves two random agents, scores goals, and resets
episodes after goals.

There is also a manual debug runner:

```bash
python scripts/play_manual.py
```

It lets a human control the left player with WASD/arrow keys and Space while the
right player remains random.

The first numeric baseline runner is:

```bash
python scripts/evaluate_agents.py --episodes 20 --seed 1
python scripts/compare_baselines.py --episodes 50 --seed 1
python scripts/play_agents.py --left safe_chase --right random --seed 1
```

These compare scripted policies and report wins, truncations, rewards, rule
violations, and ball relocations before training code is added.

Available scripted policies are currently `random`, `chase`, `safe_chase`, and
`defensive`.

## Environment

Use the repository-local virtual environment:

```bash
source .venv/bin/activate
```

Expected Python:

```text
Python 3.13.13
```

Python 3.14 caused Pygame source-build failures on macOS because SDL headers were
not available. Prefer Python 3.13 unless the dependency stack is intentionally
upgraded and revalidated.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Core Files

- `soccer_rl/envs/soccer_1v1.py`: `Soccer1v1Env`, simulation logic, rewards, and renderer
- `soccer_rl/policies.py`: scripted baseline policies
- `soccer_rl/evaluation.py`: headless evaluation metrics for baseline policies
- `scripts/evaluate_agents.py`: baseline evaluation runner
- `scripts/compare_baselines.py`: all-vs-all baseline matchup matrix runner
- `scripts/play_agents.py`: visual scripted-policy runner
- `scripts/play_random.py`: random-agent demo runner
- `scripts/play_manual.py`: human-left vs random-right debug runner
- `tests/test_soccer_env.py`: smoke tests for environment behavior and render color checks
- `tests/test_baseline_agents.py`: policy and evaluation smoke tests
- `soccer_rules.md`: human-readable current gameplay rules
- `update-log-description/v0.0.2-gameplay-training-foundation-description.md`: current v0.0.2 notes and initial baseline matrix
- `requirements.txt`: Python dependencies

## Current Behavior

`Soccer1v1Env` is a PettingZoo-style `ParallelEnv` with two agents:

- `left`
- `right`

Actions are:

```text
0 stay
1 up
2 down
3 left
4 right
5 kick
```

The renderer uses team colors consistently:

- left player and left goal: red
- right player and right goal: blue
- scoreboard panels match the same team colors

Rewards are sparse:

- scorer receives `+1.0`
- opponent receives `-1.0`
- players crossing the pale inner playable line receive `OUT_OF_BOUNDS_PENALTY`
  and are ejected for `OUT_OF_BOUNDS_EJECTION_STEPS`
- players entering either goal area box receive `GOAL_AREA_PENALTY` and the same
  ejection timeout
- other non-goal steps receive `0.0`

Most physics and render-tuning values are module-level constants near the top of
`soccer_rl/envs/soccer_1v1.py`. Prefer changing those constants before adding
new config systems. Ball movement uses inertia plus two kinds of slowdown:
`BALL_FRICTION` for proportional drag and `BALL_ROLLING_RESISTANCE` for steady
rolling resistance.

Player movement uses velocity with acceleration and deceleration. `PLAYER_SPEED`
is the maximum speed, not an instant per-step move. Opposite-direction input
resets that axis velocity before accelerating again. Active players collide as
circles; collision resolution separates overlaps and removes closing velocity.
When both robots trap the ball between them, the ball enters a contest state
instead of bouncing away; prolonged contests relocate the ball to a nearby safe
playable spot.

The pale inner field line is the playable boundary for players. Ejected players
ignore movement, kick, possession, and contest logic until their timeout expires,
then reenter on their own side.

White rectangular goal area boxes are forbidden for players but not the ball. If
a player enters one, apply the goal-area penalty and ejection timeout.

Keep `soccer_rules.md` in sync when gameplay rules or reward constants change.

## Validation

Run tests after changing environment logic or rendering:

```bash
python -m pytest
```

Run a headless random-play smoke test:

```bash
python scripts/play_random.py --no-render --episodes 2 --seed 1
```

Run a short baseline evaluation smoke test:

```bash
python scripts/evaluate_agents.py --left chase --right random --episodes 3 --seed 1 --max-cycles 120
python scripts/compare_baselines.py --episodes 2 --seed 1 --max-cycles 120
```

Compile-check scripts after changing runner code:

```bash
python -m py_compile scripts/play_random.py scripts/play_manual.py scripts/evaluate_agents.py scripts/compare_baselines.py scripts/play_agents.py
```

For renderer checks in headless contexts, use SDL's dummy video driver:

```bash
SDL_VIDEODRIVER=dummy python -c "from soccer_rl.envs import Soccer1v1Env; env=Soccer1v1Env(render_mode='rgb_array'); env.reset(seed=3); print(env.render().shape); env.close()"
```

## Development Notes

- Keep the environment small and easy to debug before adding learning code.
- Use baseline evaluation metrics before and after rule or reward changes.
- Prefer simple, deterministic tests around reset, step outputs, rewards, and rendering invariants.
- Do not commit `.venv`, cache directories, or generated bytecode.
- If rendering changes, keep UI text compact and avoid overlapping the field action.
- Preserve `python scripts/play_random.py` and `python scripts/play_manual.py` as the primary manual smoke tests.

## Git Context

The current development branch is:

```text
RA_v0.0.2-gameplay-training-foundation
```

The initialization milestone was finalized on `RA_v0.0.1-initialization`.
