from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(69))) # Nice

def crossover(pop, pop_f, p_cross):
    """Function to handle the crossover"""

    def cross(p1, p2, crossover_probability):
        if rs.rand() < crossover_probability:
            for i in range(len(p1)):
                prob = rs.rand()
                if prob <= 0.5:
                    p1[i], p2[i] = p2[i], p1[i]
        return p1, p2
    
    # We use softmax to determine weight
    exp_p = np.exp(pop_f - np.max(pop_f))
    weights = exp_p/(np.sum(exp_p)) # Higher probability to mate when parent has high score

    mating_idx = rs.choice(pop.shape[0], size=2*len(pop), p=weights, replace=True)
    mating = pop[mating_idx]

    offspring = []
    for i in range(1, len(mating), 2):
        o1, o2 = cross(mating[i-1], mating[i], crossover_probability=p_cross)
        offspring.extend([o1, o2])

    return np.array(offspring)


def mutation(offspring, p_mutate):
    """Helper function to apply the mutation"""

    p_not_mutate = 1 - p_mutate # Probability the bit wont mutate
    mutation_array = rs.choice(2, offspring.shape, replace=True, p=[p_not_mutate, p_mutate]) # Mutate according to np.abs(bit_val - 1) if mutate is true
    offspring = np.abs(offspring - mutation_array).astype(np.int64)
    return list(offspring)


def selection(parents, parents_f, offspring, offspring_f, mu_plus_lambda, top_k):
    if mu_plus_lambda:
        population, population_f = np.vstack([parents, offspring]), np.hstack([parents_f, offspring_f])
    else: 
        population, population_f = offspring, offspring_f

    best_idx = np.argsort(-np.abs(population_f))[:top_k]
    return population[best_idx], population_f[best_idx]



def s2631415_studentnumber2_GA(problem: ioh.problem.PBO, mu_plus_lambda=True, mu=10, p_crossover=0.5, mutation_r=0.02, budget=5000) -> None:

    parents = rs.choice(2, (mu, problem.bounds.lb.shape[0]), replace=True)
    parents_f = problem(parents) # type: ignore

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:

        # Generate offspring
        offspring = crossover(parents, parents_f, p_cross=p_crossover)
        offspring = mutation(offspring, mutation_r)
        
        # Evaluate offspring
        offspring_f = np.array([problem(offspring[i]) for i in range(len(offspring))])
        
        # Selection
        parents, parents_f = selection(parents, parents_f, offspring, offspring_f, mu_plus_lambda, mu)

        # Best state
        if problem.state.evaluations > 4990:
            print(f"Problem: {problem}, Evaluations: {problem.state.evaluations}, Best: {np.max(parents_f)}")
        
        # this is how you evaluate one solution `x`
        # f = problem(x)
    # no return value needed

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
        s2631415_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        s2631415_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()