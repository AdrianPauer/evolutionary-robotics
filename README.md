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

## Detailed Documentation

- [Environment](docs/environment.md): explains the world, robot state, actions, observations, reward, life system, food spawning, and rendering.
- [Fitness and evaluation](docs/fitness.md): explains rollouts, reward, life, fitness formula, training output, and quick evaluation metrics.
- [Command-line arguments](docs/arguments.md): explains every `evolution.py` argument, its default value, and examples.

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
