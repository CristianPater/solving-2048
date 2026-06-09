import numpy as np
import time
from multiprocessing import Pool
from Env2048 import Env
from GASolution.Agent import Agent


def evaluate_chromosome(args):
    chromosome, no_games = args
    agent = Agent()
    env = Env()
    agent.set_weights(chromosome)
    score = 0
    for i in range(no_games):
        env.reset()
        while not env.done:
            valid_actions = np.array([env.valid(a) for a in [env.UP, env.LEFT, env.RIGHT, env.DOWN]])
            board = np.where(env.board == 0, 0, np.log2(np.maximum(env.board, 1))).flatten()
            action = agent.act(board, valid_actions)
            env.step(action)
        score += env.score
    return score / no_games


class Ga:

    def __init__(self, n, k, t, mr, ms, g):
        self.pop_size = n
        self.no_games = k
        self.tournament_size = t
        self.mutation_rate = mr
        self.mutation_strength = ms
        self.generations = g
        self.agent = Agent()
        self.num_islands = 4
        self.island_size = n // self.num_islands
        self.islands = [self.init_population(len(self.agent.get_weights())) for _ in range(self.num_islands)]

    def random_chromosome(self, length):
        return np.random.randn(length) * 0.1

    def init_population(self, chromosome_length):
        return [self.random_chromosome(chromosome_length) for _ in range(self.island_size)]

    def tournament_selection(self, island, fitnesses):
        indices = np.random.choice(len(island), self.tournament_size, replace=False)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return island[best_idx]

    def crossover(self, parent1, parent2):
        mask = np.random.random(len(parent1)) > 0.5
        return np.where(mask, parent1, parent2)

    def mutate(self, chromosome):
        return np.where(
            np.random.random(len(chromosome)) < self.mutation_rate,
            chromosome + np.random.randn(len(chromosome)) * self.mutation_strength,
            chromosome
        )

    def migrate(self, island_fitnesses):
        for i in range(self.num_islands):
            best_idx = np.argmax(island_fitnesses[i])
            best = self.islands[i][best_idx]

            other_islands = [j for j in range(self.num_islands) if j != i]
            target = np.random.choice(other_islands)

            worst_idx = np.argmin(island_fitnesses[target])
            self.islands[target][worst_idx] = best

    def run(self):
        best_chromosome = None
        best_fitness = -np.inf
        best_fitness_history = []
        avg_fitness_history = []

        start = time.time()
        for gen in range(self.generations):
            gen_start = time.time()

            # Evaluate fitness for all islands in one pool call
            all_args = [(c, self.no_games) for island in self.islands for c in island]
            with Pool() as pool:
                all_fitnesses_flat = pool.map(evaluate_chromosome, all_args)

            # Split results back into islands
            island_fitnesses = []
            for i in range(self.num_islands):
                start_idx = i * self.island_size
                end_idx = start_idx + self.island_size
                island_fitnesses.append(all_fitnesses_flat[start_idx:end_idx])

            # Migrate every 10 generations
            if gen % 10 == 0 and gen > 0:
                self.migrate(island_fitnesses)

            # Build new populations for each island
            new_islands = []
            for i, island in enumerate(self.islands):
                fitnesses = island_fitnesses[i]
                new_population = []
                while len(new_population) < self.island_size - 1:
                    parent1 = self.tournament_selection(island, fitnesses)
                    parent2 = self.tournament_selection(island, fitnesses)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    new_population.append(child)
                # Elite
                best_idx = fitnesses.index(max(fitnesses))
                new_population.append(island[best_idx])
                new_islands.append(new_population)

            # Tracking across all islands
            all_fitnesses = [f for island_f in island_fitnesses for f in island_f]
            gen_best_fitness = max(all_fitnesses)
            gen_avg_fitness = sum(all_fitnesses) / len(all_fitnesses)

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                for i, island_f in enumerate(island_fitnesses):
                    if max(island_f) == gen_best_fitness:
                        best_chromosome = self.islands[i][island_f.index(max(island_f))]
                        break
                np.save('../best_chromosome.npy', best_chromosome)

            self.islands = new_islands


            best_fitness_history.append(gen_best_fitness)
            avg_fitness_history.append(gen_avg_fitness)

            gen_time = time.time() - gen_start
            print(
                f"Generation {gen}: best={gen_best_fitness:.1f}, avg={gen_avg_fitness:.1f}, all-time best={best_fitness:.1f}, time={gen_time:.1f}s")

        total_time = time.time() - start
        print(f"Total time: {total_time:.1f}s")
        return best_chromosome, best_fitness_history, avg_fitness_history