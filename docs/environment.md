# Environment

This document describes the `Robot2DEnv` class implemented in [env.py](/D:/skola/EvoRobotika/projekt/evolutionary-robotics/env.py:1).

## Overview

`Robot2DEnv` is a simple 2D robot environment. The robot moves inside a square world, searches for food, loses life over time, and ends the episode if it runs out of life or leaves the world.

The environment is used for:

- manual control in `env.py`
- training in `evolution.py`
- checkpoint evaluation in `evolution.py`
- rendering trained agents in `evolution.py`

## Constructor Parameters

The environment constructor is:

```python
Robot2DEnv(
    world_size=10.0,
    max_speed=3.0,
    max_steps=500,
    render_size=(800, 600),
    enable_render=False,
    spawn_in_center=False,
    food_count=6,
    obs_type="food",
)
```

Meaning of the parameters:

- `world_size`: width and height of the square world in world coordinates.
- `max_speed`: maximum forward or backward velocity magnitude.
- `max_steps`: maximum number of steps allowed in one episode.
- `render_size`: pygame window size in pixels.
- `enable_render`: whether pygame rendering is enabled.
- `spawn_in_center`: if `True`, the robot starts at `(0, 0)`. If `False`, it starts at a random position near the center.
- `food_count`: number of food pellets present at one time.
- `obs_type`: observation mode, either `food` or `ray`.

## World Geometry

The world is a square centered at `(0, 0)`.

Important geometry values:

```text
world_size = 10.0
half_size = world_size / 2
robot_radius = 0.3
food_radius = 0.2
spawn_radius = world_size * 0.15
```

With default settings:

```text
half_size = 5.0
spawn_radius = 1.5
```

The robot is considered outside the world if either coordinate exceeds the square bounds after accounting for robot radius.

## Robot State

The robot keeps these main state variables:

- `pos`: current 2D position.
- `angle`: current heading angle in radians.
- `velocity`: current signed velocity.
- `life`: remaining life.
- `food_eaten`: number of foods eaten in the current episode.
- `step_count`: number of steps taken in the current episode.
- `done`: whether the episode has ended.

At `reset()`:

- position is set either to the center or a random spawn point
- heading is randomized in `[-pi, pi]`
- velocity is set to `0`
- life is reset to `initial_life`
- food count eaten is reset to `0`
- step count is reset to `0`
- food pellets are respawned

## Dynamics

Each action is:

```text
[steer, throttle]
```

Both values are clipped to `-1..1`.

The movement constants are:

```text
steer_rate = 0.06
accel_rate = 0.05
friction = 0.95
max_speed = 3.0
```

One step does this:

1. update heading using `steer * steer_rate`
2. update velocity using `throttle * accel_rate`
3. apply friction
4. clip velocity to `[-max_speed, max_speed]`
5. move in the heading direction using `direction * velocity * 0.1`

If the next position would be outside the world, velocity is reset to `0`. The environment still marks the episode as done when the attempted new position is outside.

## Life System

Life constants:

```text
initial_life = 120
life_decay = 0.2
life_gain = 35
```

At every step:

```text
life = life - 0.2
```

When the robot touches a food pellet:

```text
life = min(initial_life, life + 35)
food_eaten += 1
```

So eating restores life, but life never exceeds `120`.

## Food Placement

Food is stored in `self.food` as a list of 2D positions.

At reset, `_spawn_foods()` creates `food_count` food pellets.

Food positions are sampled randomly inside most of the world, but not too close to the robot spawn region:

```text
uniform range: [-half_size * 0.7, half_size * 0.7]
minimum distance from spawn region: spawn_radius + food_radius + 0.5
```

When a food pellet is eaten, only that pellet is replaced by a newly sampled one.

## Observation Space

The observation depends on `obs_type`.

### `obs_type="food"`

Observation:

```text
[relative_angle, distance_norm]
```

Meaning:

- `relative_angle`: angle from the robot heading toward the nearest food, normalized by `pi` and clipped to `-0.5..0.5`
- `distance_norm`: normalized closeness to the nearest food, in `0..1`

For `distance_norm`:

- `0` means far away
- `1` means very close

### `obs_type="ray"`

Observation:

```text
[ray_0, ray_1, ray_2, ray_3, ray_4]
```

The environment casts `5` rays in a `30` degree field of view centered on the robot heading.

Ray constants:

```text
ray_count = 5
ray_range = 3.5
fov = 30 degrees
```

Each ray returns a normalized distance:

- `0` means food is very close on that ray
- `1` means no food was detected within ray range

The rays detect food only. They do not measure walls.

## Reward

Reward depends on `obs_type`.

### Reward for `obs_type="food"`

```text
reward = max(0, velocity)
```

Only forward movement is rewarded.

### Reward for `obs_type="ray"`

```text
reward = 0.01 + velocity / 20 + (1 - average_ray_distance) * 10
```

This rewards:

- a small constant survival term
- some forward movement
- being closer to visible food along the rays

## Episode Termination

The environment sets `done=True` when any of these is true:

- `life <= 0`
- the attempted new position is outside the world
- `step_count >= max_steps`

Training and evaluation in `evolution.py` also pass their own `max_steps` limit into the rollout loop, so both the environment and the outer loop now enforce the same cap.

## `step()` Return Value

`step(action)` returns:

```python
obs, reward, done, info
```

Where:

- `obs`: next observation
- `reward`: scalar step reward
- `done`: episode termination flag
- `info`: dictionary with extra values

The `info` dictionary contains:

```python
{
    "step_count": self.step_count,
    "life": self.life,
    "food_eaten": self.food_eaten,
}
```

## Rendering

If rendering is enabled, pygame shows:

- square world border
- robot body and heading line
- food pellets
- ray lines in `ray` mode
- HUD with life and step count

Ray colors:

- red: food hit inside ray range
- green: no food hit

The life bar is green-ish when life is above `40%` and red-ish when it is lower.

## Manual Demo

Running:

```powershell
python env.py
```

starts a manual demo with:

```text
spawn_in_center=True
food_count=1
obs_type="ray"
enable_render=True
max_steps=500
```

Manual controls:

- left arrow: steer left
- right arrow: steer right
- up arrow: accelerate forward
- down arrow: accelerate backward

The demo prints observation and reward values for each step.
It now also ends automatically when the environment reaches `max_steps`.
