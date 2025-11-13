from typing import List
from sobol_sampling import draw_sobol_samples
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import s2631415_studentnumber2_GA, create_problem

budget = 100000000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

# Hyperparameters to tune, e.g.

bounds = ([10, 200], # Population size
          [0.01, 0.1], # Mutation rate
          [0.1, 0.9], # Crossover rate
          [20, 200]) # offspring size

configs = draw_sobol_samples(*bounds, n_dims=4) # Draws Sobol samples from specified bounds in these dimensions

# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective
    best_score = float('inf')
    best_params = None
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    


    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                
                

    return best_params


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)