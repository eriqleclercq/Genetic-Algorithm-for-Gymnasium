import gymnasium as gym
import numpy as np
import pygad.torchga
import pygad
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import matplotlib.pyplot as plt


def fitness_func(ga_instance, solution, sol_idx):

    model_weights_dict = pygad.torchga.model_weights_as_dict(
        model=model, weights_vector=solution
    )
    model.load_state_dict(model_weights_dict)

    observation, info = env.reset()
    sum_reward = 0
    truncated = False
    num_steps = 0

    while not truncated:
        observation_tensor = torch.tensor(observation, dtype=torch.float)
        q_val = model(observation_tensor).detach().numpy()
        q_val = np.clip(q_val, action_min, action_max)
        observation, reward, done, truncated, info = env.step(q_val)
        sum_reward += reward
        num_steps += 1

    return sum_reward


def on_generation(ga_instance):
    generation = ga_instance.generations_completed

    # Print generation and fitness of the best individual
    gen = generation
    fit = ga_instance.best_solution()[1]
    print(f"\rGeneration = {gen}, Fitness = {fit}", end="", flush=True)

    # Record one run of the agent at the beginning and middle of the training
    if generation == 1 or generation == 25:
        tmp_env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
        tmp_env = RecordVideo(
            env=tmp_env, video_folder=filename, name_prefix=f"{generation}-training"
        )

        observation, _ = tmp_env.reset()
        truncated = False

        while not truncated:
            observation_tensor = torch.tensor(observation, dtype=torch.float)
            q_val = model(observation_tensor).detach().numpy()
            q_val = np.clip(q_val, action_min, action_max)
            observation, _, _, truncated, _ = tmp_env.step(q_val)
        tmp_env.close()


def myModel(observation_space, units, action_space):
    model = torch.nn.Sequential()

    model = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=observation_space,
            out_features=units,
        ),
        torch.nn.Tanh(),
        torch.nn.Linear(units, action_space),
        torch.nn.Identity(),
    )
    return model


def create_GA_instance(torch_ga, num_solutions, pst, ct, mt) -> pygad.GA:
    NUM_GEN = 50
    NUM_PARENTS_MATING = num_solutions // 2
    INIT_POP = torch_ga.population_weights
    # PARENT_SELECTION_TYPES = "sss", "rws", "rank", "tournament"
    PARENT_SELECTION_TYPE = pst
    # CROSSOVER_TYPES = "two_points", "single_point", "uniform", "scattered"
    CROSSOVER_TYPE = ct
    # MUTATION_TYPES = "random", "swap", "inversion", "scramble", "adaptive"
    MUTATION_TYPE = mt
    MUTATION_PERCENT_GENES = 30
    KEEP_PARENTS = 0
    KEEP_ELITISM = 2

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
        on_generation=on_generation,
        parallel_processing=["process", 15],
    )

    translation_table = str.maketrans({" ": None, "(": None, ")": None, ",": "_"})

    # ga_instance.summary()
    params = (
        PARENT_SELECTION_TYPE
        + "_"
        + CROSSOVER_TYPE
        + "_"
        + MUTATION_TYPE
        + "_"
        + str(num_solutions)
        + "_"
        + str(MUTATION_PERCENT_GENES).translate(translation_table)
    )
    return params, ga_instance


def main() -> None:
    global env, model, observation_size, action_min, action_max, filename

    env = gym.make("HalfCheetah-v4")

    observation_size = env.observation_space.shape[0]

    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]
    action_space = env.action_space.shape[0]

    torch.set_grad_enabled(False)

    model = myModel(
        observation_space=observation_size, units=24, action_space=action_space
    )
    print(model)

    num_solutions = 30

    solutions, solution_fits, fit_evo = [], [], []
    PARENT_SELECTION_TYPES = ["sss", "rws", "rank", "tournament"]
    CROSSOVER_TYPES = ["two_points", "single_point", "scattered"]
    MUTATION_TYPES = ["swap"]

    for parent in PARENT_SELECTION_TYPES:
        for crossover in CROSSOVER_TYPES:
            for mutation in MUTATION_TYPES:
                for _ in range(5):
                    model = myModel(
                        observation_space=observation_size, units=24, action_space=action_space
                    )
                    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=num_solutions)
                    filename, ga_instance = create_GA_instance(torch_ga, num_solutions, parent, crossover, mutation)
                    ga_instance.run()
                    env.close()

                    solution, solution_fit, _ = ga_instance.best_solution()

                    solutions.append(solution)
                    solution_fits.append(solution_fit)
                    fit_evo.append(ga_instance.best_solutions_fitness)

                # Taking best solution out of 5 runs
                solution_fits = np.array(solution_fits)
                solution = solutions[np.argmax(solution_fits)]
                mean_fit = solution_fits.mean()

                plt.figure(figsize=(10, 6))
                for i, fitness_history in enumerate(fit_evo):
                    plt.plot(fitness_history, label=f"Run {i+1}")
                plt.xlabel("Generation")
                plt.ylabel("Fitness")
                plt.title(f"{filename} with mean fitness: {mean_fit}")
                plt.legend()
                plt.savefig(filename)
                plt.show()

                #     ga_instance.run()
                #     env.close()
                #
                #     solution, solution_fitness, solution_idx = ga_instance.best_solution()
                #     print(f"Fitness value of the best solution = {solution_fitness}")
                #     print(f"Index of the best solution : {solution_idx}")


                # Load the best solution into the model
                model_weights_dict = pygad.torchga.model_weights_as_dict(
                    model=model, weights_vector=solution
                )
                model.load_state_dict(model_weights_dict)

                tmp_env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env=tmp_env, video_folder=filename)

                # play game and record the final run of the cheetah
                observation, _ = env.reset()
                sum_reward = 0

                trunctated = False
                while not trunctated:

                    observation_tensor = torch.tensor(observation, dtype=torch.float)
                    q_val = model(observation_tensor).detach().numpy()
                    q_val = np.clip(q_val, action_min, action_max)
                    observation, reward, done, trunctated, _ = env.step(q_val)
                    sum_reward += reward

                print(f"Agent achieved {sum_reward} on the final run")
                env.close()
                # ga_instance.save(filename=filename) # save the ga_instance to a pkl file


if __name__ == "__main__":
    main()
