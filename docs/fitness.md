# Fitness and Evaluation

This document explains how a robot controller is evaluated during training, quick evaluation, and visualization.

## Rollout

A rollout is one complete robot episode.

Each rollout starts with `env.reset()` and ends when one of these happens:

- robot life reaches `0`
- the robot leaves the world bounds
- the rollout reaches `--max-steps`

During a rollout, the neural network receives the current observation and returns two action values:

```text
action = [steer, throttle]
```

Both values are clipped by the environment to the range `-1..1`.

## Environment Reward

The environment computes a reward at every step. The total rollout reward is the sum of all step rewards.

The current environment uses ray-based observation only, and the step reward is:

```text
reward = 0.01 + velocity / 20 + (1 - average_ray_distance) * 10
```

Where:

- `average_ray_distance` is the average of the five ray sensor values.
- ray sensor values are normalized from `0` to `1`.
- `0` means food is very close on that ray.
- `1` means no food was detected within ray range.

This rewards being close to visible food, plus a smaller movement component.

## Life

Robot life is tracked by the environment.

```text
initial_life = 120
life_decay = 0.2
life_gain = 35
```

At reset:

```text
life = 120
food_eaten = 0
```

Every step subtracts life:

```text
life = life - 0.2
```

When the robot eats food:

```text
life = min(120, life + 35)
food_eaten = food_eaten + 1
```

The `min(120, ...)` cap means food can restore life, but life cannot go above the starting maximum.

## Fitness Formula

Evolution does not optimize raw reward directly. It optimizes a combined fitness score:

```text
fitness = total_reward + 150 * food_eaten + 0.05 * steps + 0.02 * final_life
```

Where:

- `total_reward`: sum of all environment rewards in one rollout.
- `food_eaten`: number of food pellets eaten during the rollout.
- `steps`: number of environment steps survived.
- `final_life`: remaining life at the end of the rollout.

Food has the largest direct impact. Each eaten food pellet adds `150` fitness points.

## Why Fitness Is Not Just Reward

Raw reward alone can be noisy and may reward movement without solving the survival task. The combined fitness encourages several useful behaviors at once:

- move toward food
- eat food
- survive longer
- finish with more remaining life

This makes evolution less dependent on one reward signal.

## Training Fitness

During PyGAD training, the script prints:

```text
PyGAD generation 1: best=311.77
PyGAD generation 2: best=348.45
```

`best` is the highest fitness score found in that generation.

In this example:

```text
348.45 - 311.77 = 36.68
```

So the best candidate in generation 2 improved by `36.68` fitness points over generation 1.

## Quick Evaluation

After `train-pygad` finishes, the script saves the best network and runs a fresh quick evaluation:

```text
Quick eval: fitness=..., reward=..., steps=..., food=..., life=...
```

This can differ from `Best training fitness` because it uses new rollout seeds.

The reported values are averages over multiple evaluation episodes:

- `fitness`: average combined fitness.
- `reward`: average raw total reward.
- `steps`: average number of survived steps.
- `food`: average number of food pellets eaten.
- `life`: average final remaining life.

## Reproducibility

Use `--seed` to make training or evaluation more reproducible:

```powershell
python evolution.py --mode train-pygad --seed 123
python evolution.py --mode eval --load robot_pygad_torch.npz --seed 123
```

Fitness is only directly comparable when the environment and evaluation settings are the same, especially:

- `--food-count`
- `--spawn-in-center` or `--random-spawn`
- `--episodes`
- `--max-steps`
- `--seed`
