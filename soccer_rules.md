# SoccerRL Rules

This document describes the current playable 1v1 rules for SoccerRL. The rules
are intentionally small and debug-friendly while the environment feel is still
being tuned.

## Match Format

- The current environment is 1v1 with two agents: `left` and `right`.
- The left agent attacks the right goal.
- The right agent attacks the left goal.
- A goal ends the episode and the next episode resets player and ball positions.
- Scores are kept across episodes in the demo runners.

## Actions

```text
0 stay
1 up
2 down
3 left
4 right
5 kick
```

## Scoring

- If the ball crosses the left edge inside the goal opening, `right` scores.
- If the ball crosses the right edge inside the goal opening, `left` scores.
- The scoring agent receives `+1.0`.
- The conceding agent receives `-1.0`.
- Goal rewards terminate the current episode.

## Possession And Kicking

- Players accelerate up to a maximum speed instead of moving instantly.
- Releasing movement input leaves only a very small coast before stopping.
- Pressing the opposite direction on an axis resets that axis velocity to zero
  before acceleration starts again.
- Players are circular bodies and cannot pass through each other.
- When players collide, they are separated and the closing velocity along the
  collision direction is removed.
- A free ball can be possessed by a nearby non-ejected player.
- Possession is only picked up when the ball is slow enough.
- A player carrying the ball can kick with action `5`.
- A nearby opponent can contest a carried ball without making it bounce away.
- Ejected players cannot move, kick, possess the ball, or contest.

## Ball Contests

- If both robots hold the ball between them, the ball enters a contested state.
- A contested ball is pinned between the two robot centers with zero velocity.
- While the contest remains active, neither robot owns the ball.
- If the contest lasts `BALL_CONTEST_RELOCATION_STEPS`, the ball is relocated to
  a nearby safe playable spot.
- The current timeout is `180` environment steps, about 3 seconds at 60 FPS.
- Relocation avoids goal areas and keeps clearance from both players.
- The `infos` entry marks the event with `ball_relocated`.
- Observations include the contest timer ratio.

## Playable Boundary

- The pale inner field line is the playable boundary for players.
- If a player crosses that line, the player is ejected.
- Out-of-bounds ejection lasts `OUT_OF_BOUNDS_EJECTION_STEPS`.
- The current timeout is `300` environment steps, about 5 seconds at 60 FPS.
- The player receives `OUT_OF_BOUNDS_PENALTY`.
- The current out-of-bounds penalty is `-0.20`.
- The `infos` entry marks the event with `out_of_bounds` and `ejection_steps`.

## Goal Areas

- White rectangular boxes are drawn around both goals.
- These goal areas are forbidden for players.
- The ball is allowed to enter and pass through goal areas.
- If a player enters either goal area, the player is ejected for the same timeout.
- The player receives `GOAL_AREA_PENALTY`.
- The current goal-area penalty is `-0.20`.
- The `infos` entry marks the event with `goal_area_violation` and
  `ejection_steps`.

## Ejection And Reentry

- Ejected players are hidden from the renderer during the timeout.
- Ejected players ignore movement, kick, possession, and contest logic.
- If an ejected player had possession, the ball is released.
- When the timeout ends, the player reenters on their own side outside the goal
  area box.
- Observations include ejection timer ratios for the observing agent and the
  opponent.

## Current Tuning Constants

Most rule and physics values live near the top of
`soccer_rl/envs/soccer_1v1.py`.

```text
PLAYER_SPEED = 1.00
PLAYER_ACCELERATION = 0.20
PLAYER_DECELERATION = 0.55
CONTROL_RADIUS = 2.9
TACKLE_RADIUS = 2.9
BALL_CONTEST_RADIUS = 3.2
BALL_CONTEST_RELOCATION_STEPS = 180
BALL_RELOCATION_MIN_DISTANCE = 6.0
BALL_RELOCATION_MAX_DISTANCE = 14.0
BALL_RELOCATION_PLAYER_CLEARANCE = PLAYER_RADIUS + BALL_RADIUS + 1.0
KICK_RADIUS = 4.2
KICK_SPEED = 4.2
BALL_FRICTION = 0.88
BALL_ROLLING_RESISTANCE = 0.064
OUT_OF_BOUNDS_EJECTION_STEPS = 300
OUT_OF_BOUNDS_PENALTY = -0.20
GOAL_AREA_DEPTH = 8.4
GOAL_AREA_WIDTH = DEFAULT_GOAL_WIDTH + 12.0
GOAL_AREA_PENALTY = -0.20
```

## Planned Direction

- Keep tuning the 1v1 feel toward a RoboCupJunior Soccer
  Entry/Infrared-inspired style.
- Keep the ball contestable rather than fully locked to one player.
- After the 1v1 feel is stable, expand toward 2v2 with simple team roles: one
  attacking agent and one defending agent per side.
