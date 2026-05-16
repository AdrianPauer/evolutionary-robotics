# Command-Line Arguments

All arguments are defined near the top of `evolution.py` with `parser.add_argument(...)`. The defaults shown here match the current code.

Run this to see the parser output:

```powershell
python evolution.py --help
```

## Automatic Evaluation Arguments

### `--recodex`

Default:

```text
False
```

Marks that the script is running in ReCodEx or another automatic evaluation system.

Current effect: the argument is accepted and stored in `args.recodex`, but the current code does not branch on it yet.

Example:

```powershell
python evolution.py --recodex
```

### `--render_each N`

Default:

```text
0
```

Controls rendering during evaluation rollouts.

- `0`: do not render evaluation episodes.
- `1`: render every evaluation episode.
- `2`: render every second evaluation episode.
- `5`: render every fifth evaluation episode.

This is used by quick evaluation after training and by `--mode eval`.

Example:

```powershell
python evolution.py --mode eval --load robot_pygad_torch.npz --render_each 1
```

### `--seed N`

Default:

```text
0
```

Sets the random seed used by training and evaluation.

Example:

```powershell
python evolution.py --mode train-pygad --seed 123
```

## Program Mode

### `--mode MODE`

Default:

```text
train-pygad
```

Allowed values:

- `train-pygad`: train a Torch neural network with PyGAD.
- `train-simple`: train the older NumPy neural network with the built-in genetic algorithm.
- `eval`: load a saved checkpoint and evaluate without normal visualization.
- `watch`: load a saved checkpoint and show the agent in a pygame window.

Examples:

```powershell
python evolution.py --mode train-pygad
python evolution.py --mode eval --load robot_pygad_torch.npz
python evolution.py --mode watch --load robot_pygad_torch.npz
```

## Checkpoints

### `--save PATH`

Default:

```text
robot_pygad_torch.npz
```

Path where a trained checkpoint is saved.

Example:

```powershell
python evolution.py --mode train-pygad --save my_agent.npz
```

### `--load PATH`

Default:

```text
None
```

Path to a checkpoint loaded by `--mode eval` or `--mode watch`.

Example:

```powershell
python evolution.py --mode watch --load my_agent.npz
```

## Environment Arguments

### `--obs-type TYPE`

Default:

```text
ray
```

Allowed values:

- `ray`: observation is five ray distances in a 30 degree field of view.
- `food`: observation is nearest-food relative angle and normalized distance.

For `ray`, each input is normalized from `0` to `1`. `0` means food is very close on that ray, and `1` means no food was detected within ray range.

For `food`, the observation is:

```text
[relative_angle, distance]
```

Where `relative_angle` is clipped to `-0.5..0.5`, and larger `distance` means closer food.

Example:

```powershell
python evolution.py --mode train-pygad --obs-type food
```

### `--food-count N`

Default:

```text
1
```

Number of food pellets present in the world at one time.

Example:

```powershell
python evolution.py --mode train-pygad --food-count 3
```

### `--spawn-in-center`

Default:

```text
True
```

Starts the robot at the center of the world. This is currently enabled by default.

Example:

```powershell
python evolution.py --spawn-in-center
```

### `--random-spawn`

Default:

```text
False
```

Starts the robot at a random position near the center instead of exactly at the center.

Internally, random spawn overrides center spawn:

```text
spawn_in_center = args.spawn_in_center and not args.random_spawn
```

Example:

```powershell
python evolution.py --mode train-pygad --random-spawn
```

## Network Arguments

### `--hidden-sizes SIZES`

Default:

```text
16,16
```

Comma-separated hidden layer sizes for the neural network.

Examples:

```powershell
python evolution.py --hidden-sizes 16,16
python evolution.py --hidden-sizes 32,16
python evolution.py --hidden-sizes 32,32,16
```

The output layer always has size `2`, corresponding to:

```text
[steer, throttle]
```

## Evolution Arguments

### `--population-size N`

Default:

```text
50
```

Number of candidate neural networks in each generation.

Larger populations explore more solutions but take longer per generation.

### `--generations N`

Default:

```text
30
```

Number of evolutionary generations.

More generations usually improve the best candidate, but training takes longer.

### `--episodes N`

Default:

```text
2
```

Number of rollout episodes used per candidate during training or per checkpoint during evaluation.

Higher values make fitness less random but slower.

### `--max-steps N`

Default:

```text
350
```

Maximum number of environment steps allowed in one rollout.

A rollout can still end earlier if the robot dies or leaves the world.

### `--mutation-percent-genes N`

Default:

```text
10
```

PyGAD percentage of network parameters mutated in each child.

Only used by `--mode train-pygad`.

### `--mutation-scale X`

Default:

```text
0.12
```

Maximum absolute random mutation change used by PyGAD.

With the default value, PyGAD mutations are sampled in this range:

```text
-0.12 .. 0.12
```

Only used by `--mode train-pygad`.

## Common Commands

Train with defaults:

```powershell
python evolution.py
```

Train and save to a specific checkpoint:

```powershell
python evolution.py --mode train-pygad --save robot_pygad_torch.npz
```

Evaluate a checkpoint:

```powershell
python evolution.py --mode eval --load robot_pygad_torch.npz --episodes 10 --max-steps 500
```

Watch a checkpoint:

```powershell
python evolution.py --mode watch --load robot_pygad_torch.npz --max-steps 10000
```
