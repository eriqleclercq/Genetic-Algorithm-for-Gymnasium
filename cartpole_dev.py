import time
import gymnasium as gym
import numpy as np
import pygad.torchga
import pygad
import torch
from multiprocessing import Pool


def fitness_func(ga_instance, solution, sol_idx):

    model_weights_dict = pygad.torchga.model_weights_as_dict(
        model=model, weights_vector=solution
    )
    model.load_state_dict(model_weights_dict)

    observation, info = env.reset()
    sum_reward = 0
    done = False

    while not done and sum_reward < 1000:
        observation_tensor = torch.tensor(observation, dtype=torch.float)
        q_val = model(observation_tensor)
        action = np.argmax(q_val)
        action = torch.asarray(action)
        observation, reward, done, _, _ = env.step(action.item())
        sum_reward += reward

    return sum_reward


def on_generation(ga_instance):
    print(
        "Generation = {generation}".format(generation=ga_instance.generations_completed)
    )
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def fitness_wrapper(sol):
    ga_instance = ...
    return fitness_fn(ga_instance, sol, 0)


# class PooledGA(pygad.GA):
#     def cal_pop_fitness(self):
#         global pool
#
#         pop_fitness = pool.map(fitness_wrapper, self.population)
#         print(pop_fitness)
#         pop_fitness = np.array(pop_fitness)
#         return pop_fitness


def myModel(observation_space, units, action_space):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=observation_space, out_features=units),
        torch.nn.ReLU(),
        torch.nn.Linear(units, units),
        torch.nn.ReLU(),
        torch.nn.Linear(units, action_space),
    )
    return model


def main() -> None:
    global env, model, observation_size

    env = gym.make("CartPole-v1")
    observation_size = env.observation_space.shape[0]
    action_space = env.action_space.n

    torch.set_grad_enabled(False)

    model = myModel(
        observation_space=observation_size, units=16, action_space=action_space
    )

    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=10)

    NUM_GEN = 50
    NUM_PARENTS_MATING = 5
    INIT_POP = torch_ga.population_weights
    PARENT_SELECTION_TYPE = "sss"
    CROSSOVER_TYPE = "single_point"
    MUTATION_TYPE = "random"
    MUTATION_PERCENT_GENES = 10
    KEEP_PARENTS = -1

    ga_instance = pygad.GA(
        num_generations=NUM_GEN,
        num_parents_mating=NUM_PARENTS_MATING,
        initial_population=INIT_POP,
        fitness_func=fitness_func,
        parent_selection_type=PARENT_SELECTION_TYPE,
        crossover_type=CROSSOVER_TYPE,
        mutation_type=MUTATION_TYPE,
        mutation_percent_genes=MUTATION_PERCENT_GENES,
        keep_parents=KEEP_PARENTS,
        on_generation=on_generation
    )

    ga_instance.run()

if __name__ == "__main__":
    main()
