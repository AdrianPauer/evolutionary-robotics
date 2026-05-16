# Evolution robotics project


[Slides](https://docs.google.com/presentation/d/11Ytr507oy4hYPf7N7QVgLCin32jrwfwCpaC9UYbs5gw/edit?usp=sharing)

## Setup

Install the Python packages used by the environment and evolutionary trainer:

```powershell
python -m pip install -r requirements.txt
```

`pygame-ce` provides the `pygame` module used by `env.py`. If you prefer the original `pygame` package, replace `pygame-ce` with `pygame` in `requirements.txt`.

## Train an Agent

The default trainer uses PyGAD to evolve all weights of a small Torch MLP network.

```powershell
python evolution.py --mode train-pygad --population-size 50 --generations 30 --episodes 2 --max-steps 350 --save robot_pygad_torch.npz
```

Useful options:

- `--obs-type ray` uses the ray sensor inputs. This is the default.
- `--obs-type food` uses angle and distance to the nearest food.
- `--food-count 1` sets how many food pellets exist in the world.
- `--hidden-sizes 16,16` changes the neural network hidden layers.
- `--random-spawn` trains from random starting positions instead of the center.
- `--episodes 3` evaluates each candidate on more rollouts, which is slower but less noisy.
- `--max-steps 500` lets each rollout run longer.

There is also the older built-in genetic algorithm:

```powershell
python evolution.py --mode train-simple --population-size 50 --generations 30 --episodes 2 --max-steps 350 --save robot_simple.npz
```

## Quick Evaluation

Run a saved agent without rendering and print averaged metrics:

```powershell
python evolution.py --mode eval --load robot_pygad_torch.npz --episodes 10 --max-steps 500
```

The output includes:

- `fitness`: combined score used for evolution.
- `reward`: raw environment reward.
- `steps`: average survival length.
- `food`: average food eaten.
- `life`: average final remaining robot life.

## Metrics and Terms

Training, evaluation, and watching all report values from completed rollouts. A rollout is one robot episode from environment reset until the robot dies, leaves the world, or reaches `--max-steps`.

- `fitness`: the score optimized by evolution. It combines raw reward, food eaten, survival time, and final life.
- `reward`: the sum of the environment reward over the rollout. For `--obs-type food`, each step reward is forward velocity only: `max(0, velocity)`. For `--obs-type ray`, each step reward is `0.01 + velocity / 20 + (1 - average_ray_distance) * 10`.
- `steps`: how many environment steps the robot survived in the rollout. In averaged evaluation output, this is the mean across all `--episodes`.
- `food`: how many food pellets the robot ate. In averaged evaluation output, this is the mean across all `--episodes`.
- `food_eaten`: the same value as `food` inside the code for one rollout.
- `life`: remaining robot life at the end of the rollout. In averaged evaluation output, this is the mean final life across all `--episodes`.
- `final_life`: the same value as `life` inside the code for one rollout.
- `best`: the highest fitness score found in the current generation.
- `avg`: the average fitness of the full population in the current generation. This is printed by the older `train-simple` mode.
- `Best training fitness`: the best fitness PyGAD found during training before the checkpoint was saved.
- `Quick eval`: a fresh evaluation of the saved best agent after training. This can differ from `Best training fitness` because it uses new rollout seeds.

Robot life starts at `120`. Every step subtracts `0.2` life. Eating food adds `35` life, capped at `120`. A rollout ends when life reaches `0`, when the robot leaves the world, or when `--max-steps` is reached.

## Options Reference

The command-line parser is defined near the top of `evolution.py`. Default values are visible directly in the `parser.add_argument(...)` calls, and every value can be overridden when starting the program with the matching CLI flag.

- `--recodex`: mark that the script is running in ReCodEx or another automatic evaluation system.
- `--render_each N`: render every N evaluation episodes; `0` disables evaluation rendering.
- `--mode train-pygad`: train a Torch neural network with PyGAD. This is the default mode.
- `--mode train-simple`: train the older NumPy `MLPBrain` with the built-in genetic algorithm.
- `--mode eval`: run a saved checkpoint without rendering and print averaged metrics.
- `--mode watch`: run a saved checkpoint with pygame rendering.
- `--save PATH`: checkpoint path written after training.
- `--load PATH`: checkpoint path used by `eval` or `watch`.
- `--obs-type ray`: observation is five ray distances in a 30 degree field of view. Each value is normalized from `0` to `1`; `0` means food is very close on that ray, and `1` means no food was detected within ray range.
- `--obs-type food`: observation is `[relative_angle, distance]` for the nearest food. `relative_angle` is normalized by pi and clipped to `-0.5..0.5`; `distance` is normalized so larger values mean closer food.
- `--food-count N`: number of food pellets present at once.
- `--spawn-in-center`: start the robot at the center. This is currently the default.
- `--random-spawn`: start the robot near the center with random position instead of exactly at the center.
- `--hidden-sizes 16,16`: hidden layer sizes for the neural network.
- `--population-size N`: number of candidate networks in each generation.
- `--generations N`: number of evolutionary generations to run.
- `--episodes N`: number of rollout episodes used per candidate or evaluation. Higher values are slower but less random.
- `--max-steps N`: maximum steps allowed per rollout.
- `--seed N`: random seed used for reproducible training or evaluation.
- `--mutation-percent-genes N`: PyGAD percentage of network parameters mutated per child.
- `--mutation-scale X`: maximum size of PyGAD random mutation changes.

## Fitness Meaning

During training, lines like this:

```text
PyGAD generation 1: best=311.77
PyGAD generation 2: best=348.45
```

mean that PyGAD found a better candidate network in generation 2 than in generation 1. The `best` value is the highest fitness score in that generation.

The current fitness formula is:

```text
fitness = reward + 150 * food_eaten + 0.05 * steps + 0.02 * final_life
```

So a higher value usually means the robot moved well, survived longer, finished with more life, and especially ate more food. Eating food has the largest direct impact because each food pellet adds `150` fitness points.

Example interpretation:

- `best=311.77`: the best robot in generation 1 scored `311.77`.
- `best=348.45`: the best robot in generation 2 scored `348.45`, so evolution improved the best candidate by `36.68` points.

The score is only comparable between runs when the environment settings are the same, especially `--obs-type`, `--food-count`, `--episodes`, `--max-steps`, and spawn mode.

## Manual Visualization

Run the environment manually with keyboard control:

```powershell
python env.py
```

Controls:

- `Left Arrow`: steer left.
- `Right Arrow`: steer right.
- `Up Arrow`: accelerate forward.
- `Down Arrow`: accelerate backward.

The manual demo currently uses `obs_type='ray'`, one food pellet, and starts the robot in the center.

## Agent Visualization

Watch a trained agent in the pygame window:

```powershell
python evolution.py --mode watch --load robot_pygad_torch.npz --max-steps 10000
```

Use the same environment options as training if the checkpoint was trained with different settings. For example:

```powershell
python evolution.py --mode watch --load robot_food.npz --obs-type food --food-count 3 --max-steps 10000
```
