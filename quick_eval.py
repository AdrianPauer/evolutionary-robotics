import argparse
import statistics

import numpy as np

from evolution import evaluate_brain_report, load_brain, make_env, parse_hidden_sizes, rollout_brain, torch


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint across multiple seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--load", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--episodes", default=1, type=int, help="Episodes per seed.")
    parser.add_argument("--max-steps", default=500, type=int, help="Maximum steps per episode.")
    parser.add_argument("--seed-start", default=0, type=int, help="First seed to evaluate.")
    parser.add_argument("--seed-count", default=10, type=int, help="How many consecutive seeds to evaluate.")
    parser.add_argument("--food-count", default=1, type=int, help="Number of food pellets in the world.")
    parser.add_argument("--spawn-in-center", default=True, action="store_true", help="Start the robot at the center.")
    parser.add_argument("--random-spawn", default=False, action="store_true", help="Start the robot at a random position near the center.")
    parser.add_argument("--hidden-sizes", default="16,16", help="Comma-separated hidden layer sizes for old checkpoints without metadata.")
    return parser


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def format_summary(name, values):
    stats = summarize(values)
    return (
        f"{name:<7}"
        f" mean={stats['mean']:.2f}"
        f" std={stats['std']:.2f}"
        f" min={stats['min']:.2f}"
        f" max={stats['max']:.2f}"
    )


def main():
    args = build_parser().parse_args()
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    env_kwargs = {
        "spawn_in_center": args.spawn_in_center and not args.random_spawn,
        "food_count": args.food_count,
        "enable_render": False,
        "vision_mask": False,
        "obs_type": "ray",
    }

    probe_env = make_env(env_kwargs, enable_render=False)
    input_size = len(probe_env.reset())
    probe_env.close()

    brain = load_brain(args.load, input_size=input_size, hidden_sizes=hidden_sizes)

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        if args.episodes == 1:
            np.random.seed(seed)
            if torch is not None:
                torch.manual_seed(seed)
            env = make_env(env_kwargs, enable_render=False)
            result = rollout_brain(brain, env, max_steps=args.max_steps, render=False)
            env.close()
            row = {
                "seed": seed,
                "fitness": result.fitness,
                "reward": result.total_reward,
                "steps": float(result.steps),
                "food": float(result.food_eaten),
                "life": result.final_life,
            }
        else:
            report = evaluate_brain_report(
                brain,
                env_kwargs,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=seed,
                render_each=0,
            )
            row = {
                "seed": seed,
                "fitness": report["fitness"],
                "reward": report["reward"],
                "steps": report["steps"],
                "food": report["food"],
                "life": report["life"],
            }
        rows.append(row)
        print(
            f"seed={row['seed']:>4} "
            f"fitness={row['fitness']:.2f} "
            f"reward={row['reward']:.2f} "
            f"steps={row['steps']:.1f} "
            f"food={row['food']:.2f} "
            f"life={row['life']:.1f}"
        )

    print()
    print("Summary")
    print(format_summary("fitness", [row["fitness"] for row in rows]))
    print(format_summary("reward", [row["reward"] for row in rows]))
    print(format_summary("steps", [row["steps"] for row in rows]))
    print(format_summary("food", [row["food"] for row in rows]))
    print(format_summary("life", [row["life"] for row in rows]))

    best_row = max(rows, key=lambda row: row["fitness"])
    worst_row = min(rows, key=lambda row: row["fitness"])
    print()
    print(f"best_seed  seed={best_row['seed']} fitness={best_row['fitness']:.2f}")
    print(f"worst_seed seed={worst_row['seed']} fitness={worst_row['fitness']:.2f}")


if __name__ == "__main__":
    main()
