import numpy as np

from env import Robot2DEnv


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
        np.savez(filename, params=self.get_parameters())

    def load(self, filename):
        data = np.load(filename)
        self.set_parameters(data["params"])

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
        assert len(a_params) == len(b_params), "Parent networks must have the same architecture"
        mask = np.random.rand(len(a_params)) < 0.5
        child = parent_a.copy()
        child.set_parameters(np.where(mask, a_params, b_params))
        return child


def evaluate_brain(brain, env, max_steps=800, render=False):
    obs = env.reset()
    total_reward = 0.0
    step = 0

    while True:
        action = brain.forward(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        if render:
            env.render()

        if done or step >= max_steps:
            break
    
    # fitness_food_global = (
    #     total_reward / 10.0
    #     + 10 * info.get("food_eaten", 0)
    #     + step
    # )

    
    #print(f"Evaluation: total_reward={total_reward:.2f}, food_eaten={info.get('food_eaten', 0)}, steps={step}")

    return total_reward


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
):
    test_obs = env.reset()
    print(f"Observation: {test_obs}")

    input_size = len(test_obs)
    population = create_population(population_size, input_size, hidden_sizes)

    elite_count = max(1, int(population_size * elite_fraction))
    best_brain = None
    best_fitness = -np.inf

    for generation in range(1, generations + 1):
        fitnesses = [evaluate_brain(brain, env) for brain in population]
        order = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in order]
        fitnesses = [fitnesses[i] for i in order]

        if fitnesses[0] > best_fitness:
            best_fitness = fitnesses[0]
            best_brain = population[0].copy()

        print(f"Generation {generation}: best={fitnesses[0]:.2f}, avg={np.mean(fitnesses):.2f}")

        next_population = [population[i].copy() for i in range(elite_count)]

        while len(next_population) < population_size:
            parent_a = tournament_select(population, fitnesses, tournament_size)
            parent_b = tournament_select(population, fitnesses, tournament_size)
            child = MLPBrain.crossover(parent_a, parent_b)
            child.mutate(mutation_rate=mutation_rate, mutation_scale=mutation_scale)
            next_population.append(child)

        population = next_population

    return best_brain


def watch_brain(brain, env, max_steps=10000):
    obs = env.reset()
    env.enable_render = True
    env._init_pygame()
    done = False
    step = 0

    while not done and step < max_steps:
        action = brain.forward(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        step += 1

    env.close()
    return info


if __name__ == "__main__":
    environment = Robot2DEnv(spawn_in_center=True, food_count=1, enable_render=False, obs_type='ray')
    # best = evolve(
    #     env=environment,
    #     population_size=50,
    #     generations=30,
    #     hidden_sizes=(16, 16),
    # )
    save_file = 'robot_ray.npz'

    print("Evolution finished. Watching best agent...")
    
    # best.save(save_file)
    # #evaluate_brain(best, environment, render=True)
    
    test_obs = environment.reset()
    input_size = len(test_obs)
    
    best = MLPBrain(input_size, hidden_sizes=(16, 16), output_size=2)
    best.load(save_file)

    watch_environment = Robot2DEnv(spawn_in_center=True, food_count=1, enable_render=True, obs_type='ray')
    watch_brain(best, watch_environment)
