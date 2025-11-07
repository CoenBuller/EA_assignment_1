from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

budget = 5000
rs = RandomState(MT19937(SeedSequence(69))) # Nice

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`


def crossover(pop, pop_f, n_offspring, n_crossovers):
    """Function to handle the crossover"""

    def cross(parent1, parent2, n_cross):
        """Small helper function to do the crossover between two parents"""
        child = np.zeros_like(parent1)
        split_idx = np.linspace(0, len(parent1), n_cross, dtype=np.int8)[1:]

        for i in range(len(split_idx)):
            idx = split_idx[i]
            prev_idx = split_idx[i-1]

            choice = rs.choice(2, 1)
            if choice == 0: # If 0, we choose parent1 -> else parent 2
                child[prev_idx: idx] = parent1[prev_idx: idx]
            else:
                child[prev_idx: idx] = parent2[prev_idx: idx]
                
        return child
    
    # We use softmax to determine weight
    exp_p = np.exp(pop_f)
    weights = exp_p/(np.sum(exp_p)) # Higher probability to mate when parent has high score

    mating = rs.choice(pop, size=2*n_offspring, p=weights, replace=True)
    offspring = []
    for i in range(1, len(mating), 2):
        offspring[i] = cross(pop[i-1], pop[i], n_cross=n_crossovers)
    
    return offspring


def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    # initial_pop = ... make sure you randomly create the first population

    MU = 10 # Population size
    LAMBDA = 20 # Offspring size

    pop = rs.choice(2, (MU, problem.bounds.lb.shape[0]), replace=True)
    pop_f = problem(pop)

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:


        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
    # no return value needed
        pass


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()