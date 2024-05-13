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
    num_steps = 0

    while not done and num_steps < 5000:
        observation_tensor = torch.tensor(observation, dtype=torch.float)
        q_val = model(observation_tensor).detach().numpy()
        q_val = np.clip(q_val, action_min, action_max)
        # action = np.argmax(q_val)
        # action = torch.asarray(action)
        observation, reward, done, _, _ = env.step(q_val)
        sum_reward += reward
        num_steps +=1

    return sum_reward


def on_generation(ga_instance):
    if ga_instance.generations_completed % 2 == 0:
        # print(
        #     "Generation = {generation}".format(generation=ga_instance.generations_completed)
        # )
        # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
        gen = ga_instance.generations_completed
        fit = ga_instance.best_solution()[1]
        print(
            f"\rGeneration = {gen}, Fitness = {fit}",
            end="", flush=True
        )


def myModel(observation_space, units, action_space):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=observation_space, out_features=units),
        torch.nn.Tanh(),
        torch.nn.Linear(units, units),
        torch.nn.Tanh(),
        torch.nn.Linear(units, action_space),
        torch.nn.Identity()
    )
    return model


def main() -> None:
    global env, model, observation_size, action_min, action_max

    env = gym.make("HalfCheetah-v4")
    observation_size = env.observation_space.shape[0]

    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]
    action_space = env.action_space.shape[0]

    torch.set_grad_enabled(False)

    model = myModel(
        observation_space=observation_size, units=16, action_space=action_space
    )

    num_solutions = 10
    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=num_solutions)

    NUM_GEN = 50
    NUM_PARENTS_MATING = num_solutions
    INIT_POP = torch_ga.population_weights
    PARENT_SELECTION_TYPE = "sss"
    CROSSOVER_TYPE = "two_points"
    MUTATION_TYPE = "random"
    MUTATION_PERCENT_GENES = 10
    KEEP_PARENTS = -1
    KEEP_ELITISM = 1

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
        keep_elitism=KEEP_ELITISM,
        on_generation=on_generation
    )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))


    env = gym.make("HalfCheetah-v4", render_mode="human")
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)

    # play game
    observation, _ = env.reset()
    sum_reward = 0
    done = False
    max_steps = 1000
    current_step = 0
    filename="test_save"

    while current_step < max_steps:
        env.render()
        observation_tensor = torch.tensor(observation, dtype=torch.float)
        q_val = model(observation_tensor).detach().numpy()
        q_val = np.clip(q_val, action_min, action_max)
        observation, reward, done, _, _ = env.step(q_val)
        sum_reward += reward
        current_step += 1
        # action = np.argmax(q_val)
        # action = torch.asarray(action)
        # observation_tensor = torch.tensor(observation, dtype=torch.float)
        # q_val = model(observation_tensor)
        # action = np.argmax(q_val)
        # action = torch.asarray(action)
        # observation, reward, done, _, _ = env.step(action.item())
        # sum_reward += reward
    ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
    ga_instance.save(filename=filename)
    print("DONE!")

if __name__ == "__main__":
    main()
