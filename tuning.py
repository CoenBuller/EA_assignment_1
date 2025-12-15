from typing import List
from sobol_sampling import draw_sobol_samples
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import s2631415_studentnumber2_GA, create_problem, initialize
from ES import student4398270, create_problem

seed = 69
budget = 100000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

# Hyperparameters to tune, e.g.

bounds = (
          [0.02, 0.1], # Mutation rate
          [0.1, 0.5] # Crossover probability
          ) 

configs = draw_sobol_samples(*bounds, n_dims=4) # Draws Sobol samples from specified bounds in these dimensions

# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective
    best_score = -float('inf')
    best_params = []
    # create the LABS problem and the data logger
    F18, _logger1 = create_problem(dimension=50, fid=18)
    # create the N-Queens problem and the data logger
    F23, _logger2 = create_problem(dimension=49, fid=23)

    initial_pop18 = initialize(mu=250, problem=F18)
    initial_pop23 = initialize(mu=250, problem=F23)

    for config in configs:
        mu, p_mutate, crossover_r = config
        _, initial_max18, best_score18 = s2631415_studentnumber2_GA(problem=F18, mu_plus_lambda=True, mu=mu, p_crossover=crossover_r, mutation_r=p_mutate, budget=5000, initial_pop=initial_pop18)
        _, initial_max23, best_score23 = s2631415_studentnumber2_GA(problem=F23,  mu_plus_lambda=True, mu=mu, p_crossover=crossover_r, mutation_r=p_mutate, budget=5000, initial_pop=initial_pop23) 

        # These scores measure with what factor the fitness increased relative to maximum fitness at initialization
        score18 = abs((best_score18 - initial_max18)/initial_max18)
        score23 = abs((best_score23 - initial_max23)/initial_max23)
        total_score = score18 + score23 # The total score is just the sums of the individual scores of F18 and F23
        if total_score > best_score:
            best_score, best_params = total_score, config

    return best_params


if __name__ == "__main__":

    # Hyperparameter tuning to determine the best parameters for both problems
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)