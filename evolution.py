import argparse
import os
from datetime import datetime
from dataclasses import dataclass

import numpy as np

try:
    import pygad
except ImportError:
    pygad = None

try:
    import torch
    from torch import nn
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
except ImportError:
    torch = None
    nn = None
    parameters_to_vector = None
    vector_to_parameters = None

from env import Robot2DEnv


parser = argparse.ArgumentParser(
    description="Train and evaluate robot brains.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx.")
parser.add_argument("--render_each", default=0, type=int, help="Render every N evaluation episodes; 0 disables rendering.")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument("--mode", choices=("train-pygad", "train-simple", "eval", "watch"), default="train-pygad", help="Program mode.")
parser.add_argument("--save", default="robot_pygad_torch.npz", help="Checkpoint path written after training.")
parser.add_argument("--load", default=None, help="Checkpoint path loaded by eval or watch mode.")
parser.add_argument("--food-count", default=1, type=int, help="Number of food pellets in the world.")
parser.add_argument("--spawn-in-center", default=True, action="store_true", help="Start the robot at the center.")
parser.add_argument("--random-spawn", default=False, action="store_true", help="Start the robot at a random position near the center.")
parser.add_argument("--hidden-sizes", default="16,16", help="Comma-separated hidden layer sizes.")
parser.add_argument("--population-size", default=50, type=int, help="Number of candidate networks per generation.")
parser.add_argument("--generations", default=30, type=int, help="Number of evolutionary generations.")
parser.add_argument("--episodes", default=2, type=int, help="Rollout episodes per candidate or evaluation.")
parser.add_argument("--max-steps", default=800, type=int, help="Maximum steps per rollout.")
parser.add_argument("--checkpoint-every", default=10, type=int, help="Save a checkpoint every N generations; 0 disables periodic checkpoints.")
parser.add_argument("--mutation-percent-genes", default=10, type=int, help="Percentage of genes mutated by PyGAD.")
parser.add_argument("--mutation-scale", default=0.12, type=float, help="Maximum absolute PyGAD mutation change.")


class MLPBrain:
    def __init__(self, input_size, hidden_sizes=(16, 16), output_size=2, params=None):
        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.output_size = output_size
        self.layer_sizes = [input_size] + list(self.hidden_sizes) + [output_size]
        self.params = params if params is not None else self._init_params()

    def _init_params(self):
        params = []
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            w = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(2.0 / (in_size + out_size))
            b = np.zeros(out_size, dtype=np.float32)
            params.append((w, b))
        return params

    def forward(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for w, b in self.params[:-1]:
            x = np.tanh(np.dot(x, w) + b)
        w, b = self.params[-1]
        return np.tanh(np.dot(x, w) + b)

    def get_parameters(self):
        return np.concatenate([p.ravel() for layer in self.params for p in layer])

    def set_parameters(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        offset = 0
        new_params = []
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weight_size = in_size * out_size
            w = vector[offset:offset + weight_size].reshape((in_size, out_size))
            offset += weight_size
            b = vector[offset:offset + out_size]
            offset += out_size
            new_params.append((w.astype(np.float32), b.astype(np.float32)))
        self.params = new_params

    def save(self, filename):
        np.savez(
            filename,
            params=self.get_parameters(),
            brain_type="numpy",
            input_size=self.input_size,
            hidden_sizes=np.asarray(self.hidden_sizes, dtype=np.int32),
            output_size=self.output_size,
        )

    @property
    def param_count(self):
        return sum(w.size + b.size for w, b in self.params)

    def copy(self):
        copied = [(w.copy(), b.copy()) for w, b in self.params]
        return MLPBrain(self.input_size, self.hidden_sizes, self.output_size, params=copied)

    def mutate(self, mutation_rate=0.05, mutation_scale=0.1):
        vector = self.get_parameters()
        mask = np.random.rand(len(vector)) < mutation_rate
        vector[mask] += np.random.randn(mask.sum()).astype(np.float32) * mutation_scale
        self.set_parameters(vector)

    @staticmethod
    def crossover(parent_a, parent_b):
        a_params = parent_a.get_parameters()
        b_params = parent_b.get_parameters()
        mask = np.random.rand(len(a_params)) < 0.5
        child = parent_a.copy()
        child.set_parameters(np.where(mask, a_params, b_params))
        return child


class TorchMLPBrain(nn.Module):
    def __init__(self, input_size, hidden_sizes=(16, 16), output_size=2):
        if torch is None:
            raise ImportError("torch is required for TorchMLPBrain. Install it with 'pip install torch'.")

        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.output_size = output_size

        layers = []
        prev_size = input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        self.eval()
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_tensor(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32)
        return self.net(x)

    def forward(self, obs):
        with torch.no_grad():
            return self.forward_tensor(obs).cpu().numpy()

    def get_parameters(self):
        with torch.no_grad():
            return parameters_to_vector(self.parameters()).detach().cpu().numpy().astype(np.float32)

    def set_parameters(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        tensor = torch.as_tensor(vector, dtype=torch.float32)
        vector_to_parameters(tensor, self.parameters())

    def copy(self):
        copied = TorchMLPBrain(self.input_size, self.hidden_sizes, self.output_size)
        copied.set_parameters(self.get_parameters())
        return copied

    def save(self, filename):
        np.savez(
            filename,
            params=self.get_parameters(),
            brain_type="torch",
            input_size=self.input_size,
            hidden_sizes=np.asarray(self.hidden_sizes, dtype=np.int32),
            output_size=self.output_size,
        )

    def load(self, filename):
        data = np.load(filename)
        self.set_parameters(data["params"])

    @property
    def param_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))


@dataclass
class RolloutResult:
    total_reward: float
    fitness: float
    steps: int
    food_eaten: int
    final_life: float


def env_kwargs_from_env(env, enable_render=False):
    return {
        "world_size": env.world_size,
        "max_speed": env.max_speed,
        "max_steps": env.max_steps,
        "render_size": env.render_size,
        "enable_render": enable_render,
        "spawn_in_center": env.spawn_in_center,
        "food_count": env.food_count,
        "obs_type": env.obs_type,
    }


def make_env(env_kwargs=None, **overrides):
    kwargs = {} if env_kwargs is None else dict(env_kwargs)
    kwargs.update(overrides)
    return Robot2DEnv(**kwargs)


def score_rollout(total_reward, steps, info):
    food_eaten = info.get("food_eaten", 0)
    final_life = info.get("life", 0.0)
    return total_reward + 150.0 * food_eaten + 0.05 * steps + 0.02 * final_life


def rollout_brain(brain, env, max_steps=800, render=False):
    obs = env.reset()
    total_reward = 0.0
    step = 0
    info = {"life": getattr(env, "life", 0.0), "food_eaten": 0}

    while True:
        action = brain.forward(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        if render:
            env.render()

        if done or step >= max_steps:
            break

    return RolloutResult(
        total_reward=float(total_reward),
        fitness=float(score_rollout(total_reward, step, info)),
        steps=step,
        food_eaten=int(info.get("food_eaten", 0)),
        final_life=float(info.get("life", 0.0)),
    )


def evaluate_brain(brain, env=None, max_steps=800, render=False, episodes=1, seed=None, env_kwargs=None):
    if env is None and env_kwargs is None:
        raise ValueError("Pass either env or env_kwargs.")

    scores = []
    for episode in range(episodes):
        if seed is not None:
            np.random.seed(seed + episode)
            if torch is not None:
                torch.manual_seed(seed + episode)

        current_env = env if env is not None else make_env(env_kwargs, enable_render=render)
        result = rollout_brain(brain, current_env, max_steps=max_steps, render=render)
        scores.append(result.fitness)

        if env is None:
            current_env.close()

    return float(np.mean(scores))


def evaluate_brain_report(brain, env_kwargs, episodes=5, max_steps=800, seed=0, render_each=0):
    results = []
    for episode in range(episodes):
        np.random.seed(seed + episode)
        render = render_each > 0 and episode % render_each == 0
        env = make_env(env_kwargs, enable_render=render)
        results.append(rollout_brain(brain, env, max_steps=max_steps, render=render))
        env.close()

    return {
        "fitness": float(np.mean([result.fitness for result in results])),
        "reward": float(np.mean([result.total_reward for result in results])),
        "steps": float(np.mean([result.steps for result in results])),
        "food": float(np.mean([result.food_eaten for result in results])),
        "life": float(np.mean([result.final_life for result in results])),
    }


def create_population(pop_size, input_size, hidden_sizes=(16, 16), output_size=2):
    return [MLPBrain(input_size, hidden_sizes, output_size) for _ in range(pop_size)]


def tournament_select(population, fitnesses, tournament_size=3):
    indices = np.random.choice(len(population), size=tournament_size, replace=False)
    best_index = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_index]


def evolve(
    env,
    population_size=40,
    generations=25,
    elite_fraction=0.2,
    mutation_rate=0.1,
    mutation_scale=0.08,
    tournament_size=3,
    hidden_sizes=(16, 16),
    episodes=2,
    max_steps=500,
    checkpoint_path=None,
    checkpoint_every=10,
):
    test_obs = env.reset()
    print(f"Observation: {test_obs}")

    input_size = len(test_obs)
    population = create_population(population_size, input_size, hidden_sizes)
    env_kwargs = env_kwargs_from_env(env)

    elite_count = max(1, int(population_size * elite_fraction))
    best_brain = None
    best_fitness = -np.inf

    for generation in range(1, generations + 1):
        fitnesses = [
            evaluate_brain(
                brain,
                env_kwargs=env_kwargs,
                episodes=episodes,
                max_steps=max_steps,
                seed=generation * 10000 + index * 100,
            )
            for index, brain in enumerate(population)
        ]
        order = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in order]
        fitnesses = [fitnesses[i] for i in order]

        if fitnesses[0] > best_fitness:
            best_fitness = fitnesses[0]
            best_brain = population[0].copy()

        print(f"[{current_timestamp()}] Generation {generation}: best={fitnesses[0]:.2f}, avg={np.mean(fitnesses):.2f}")

        if checkpoint_path and checkpoint_every > 0 and generation % checkpoint_every == 0:
            save_generation_checkpoint(best_brain, checkpoint_path, generation)

        next_population = [population[i].copy() for i in range(elite_count)]

        while len(next_population) < population_size:
            parent_a = tournament_select(population, fitnesses, tournament_size)
            parent_b = tournament_select(population, fitnesses, tournament_size)
            child = MLPBrain.crossover(parent_a, parent_b)
            child.mutate(mutation_rate=mutation_rate, mutation_scale=mutation_scale)
            next_population.append(child)

        population = next_population

    return best_brain


def evolve_pygad(
    env_kwargs,
    population_size=50,
    generations=30,
    hidden_sizes=(16, 16),
    episodes=2,
    max_steps=350,
    seed=0,
    mutation_percent_genes=10,
    mutation_scale=0.12,
    checkpoint_path=None,
    checkpoint_every=10,
):
    if pygad is None:
        raise ImportError("pygad is required. Install it with 'pip install pygad'.")

    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    probe_env = make_env(env_kwargs, enable_render=False)
    input_size = len(probe_env.reset())
    probe_env.close()

    brain = TorchMLPBrain(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2)
    num_genes = brain.param_count

    def fitness_func(ga_instance, solution, solution_idx):
        brain.set_parameters(solution)
        rollout_seed = seed + solution_idx * 1000
        return evaluate_brain(
            brain,
            env_kwargs=env_kwargs,
            episodes=episodes,
            max_steps=max_steps,
            seed=rollout_seed,
        )

    def on_generation(ga_instance):
        solution, solution_fitness, _ = ga_instance.best_solution()
        generation = ga_instance.generations_completed
        print(f"[{current_timestamp()}] PyGAD generation {generation}: best={solution_fitness:.2f}")
        if checkpoint_path and checkpoint_every > 0 and generation % checkpoint_every == 0:
            checkpoint_brain = TorchMLPBrain(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2)
            checkpoint_brain.set_parameters(solution)
            save_generation_checkpoint(checkpoint_brain, checkpoint_path, generation)

    ga_instance = pygad.GA(
        num_generations=generations,
        sol_per_pop=population_size,
        num_parents_mating=max(2, population_size // 4),
        num_genes=num_genes,
        fitness_func=fitness_func,
        init_range_low=-0.75,
        init_range_high=0.75,
        parent_selection_type="tournament",
        K_tournament=3,
        keep_elitism=max(1, population_size // 10),
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=mutation_percent_genes,
        mutation_by_replacement=False,
        random_mutation_min_val=-mutation_scale,
        random_mutation_max_val=mutation_scale,
        random_seed=seed,
        suppress_warnings=True,
        on_generation=on_generation,
    )
    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    best_brain = TorchMLPBrain(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2)
    best_brain.set_parameters(solution)
    return best_brain, float(solution_fitness), ga_instance


def load_brain(filename, input_size=None, hidden_sizes=(16, 16), output_size=2, brain_type=None):
    data = np.load(filename)
    saved_type = str(data["brain_type"]) if "brain_type" in data.files else "numpy"
    resolved_type = brain_type or saved_type
    resolved_input_size = int(data["input_size"]) if "input_size" in data.files else input_size
    resolved_hidden_sizes = tuple(data["hidden_sizes"].tolist()) if "hidden_sizes" in data.files else tuple(hidden_sizes)
    resolved_output_size = int(data["output_size"]) if "output_size" in data.files else output_size

    if resolved_input_size is None:
        raise ValueError("input_size is required for old checkpoints without metadata.")

    if resolved_type == "torch":
        brain = TorchMLPBrain(resolved_input_size, resolved_hidden_sizes, resolved_output_size)
    else:
        brain = MLPBrain(resolved_input_size, resolved_hidden_sizes, resolved_output_size)
    brain.set_parameters(data["params"])
    return brain


def watch_brain(brain, env, max_steps=10000):
    env.enable_render = True
    env._init_pygame()
    result = rollout_brain(brain, env, max_steps=max_steps, render=True)
    env.close()
    return result


def parse_hidden_sizes(value):
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def checkpoint_filename(base_path, generation):
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".npz"
    return f"{root}_gen_{generation:04d}{ext}"


def save_generation_checkpoint(brain, base_path, generation):
    checkpoint_path = checkpoint_filename(base_path, generation)
    brain.save(checkpoint_path)
    print(f"[{current_timestamp()}] Saved checkpoint: {checkpoint_path}")


def main():
    args = parser.parse_args()
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    env_kwargs = {
        "spawn_in_center": args.spawn_in_center and not args.random_spawn,
        "food_count": args.food_count,
        "enable_render": False,
        "obs_type": "ray",
    }
    probe_env = make_env(env_kwargs, enable_render=False)
    input_size = len(probe_env.reset())
    probe_env.close()

    if args.mode == "train-pygad":
        best, fitness, _ = evolve_pygad(
            env_kwargs=env_kwargs,
            population_size=args.population_size,
            generations=args.generations,
            hidden_sizes=hidden_sizes,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            mutation_percent_genes=args.mutation_percent_genes,
            mutation_scale=args.mutation_scale,
            checkpoint_path=args.save,
            checkpoint_every=args.checkpoint_every,
        )
        best.save(args.save)
        report = evaluate_brain_report(
            best,
            env_kwargs,
            episodes=max(5, args.episodes),
            max_steps=args.max_steps,
            seed=args.seed + 50000,
            render_each=args.render_each,
        )
        print(f"Saved {args.save}")
        print(f"Best training fitness: {fitness:.2f}")
        print(
            "Quick eval: "
            f"fitness={report['fitness']:.2f}, reward={report['reward']:.2f}, "
            f"steps={report['steps']:.1f}, food={report['food']:.2f}, life={report['life']:.1f}"
        )

    elif args.mode == "train-simple":
        env = make_env(env_kwargs)
        best = evolve(
            env=env,
            population_size=args.population_size,
            generations=args.generations,
            hidden_sizes=hidden_sizes,
            episodes=args.episodes,
            max_steps=args.max_steps,
            checkpoint_path=args.save,
            checkpoint_every=args.checkpoint_every,
        )
        env.close()
        best.save(args.save)
        print(f"Saved {args.save}")

    elif args.mode == "eval":
        if args.load is None:
            raise ValueError("--load is required for eval mode.")
        brain = load_brain(args.load, input_size=input_size, hidden_sizes=hidden_sizes)
        report = evaluate_brain_report(
            brain,
            env_kwargs,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            render_each=args.render_each,
        )
        print(
            f"fitness={report['fitness']:.2f}, reward={report['reward']:.2f}, "
            f"steps={report['steps']:.1f}, food={report['food']:.2f}, life={report['life']:.1f}"
        )

    elif args.mode == "watch":
        if args.load is None:
            raise ValueError("--load is required for watch mode.")
        brain = load_brain(args.load, input_size=input_size, hidden_sizes=hidden_sizes)
        watch_env = make_env(env_kwargs, enable_render=True)
        result = watch_brain(brain, watch_env, max_steps=args.max_steps)
        print(
            f"watch: fitness={result.fitness:.2f}, reward={result.total_reward:.2f}, "
            f"steps={result.steps}, food={result.food_eaten}, life={result.final_life:.1f}"
        )


if __name__ == "__main__":
    main()
