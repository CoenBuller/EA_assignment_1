import ioh
from ioh import get_problem, logger, ProblemClass
import sys
import numpy as np
import time
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import typing

rs = RandomState(MT19937(SeedSequence(69))) # Nice

def discrete_recombination(parents):
    """
    Uniform recombination - for each bit randomly pick from the parents
    """
    mu, n = parents.shape
    offspring = np.zeros(n, dtype=int)
    for i in range(n):
        offspring[i] = rs.choice(parents[:, i])
    return offspring

def student4398270(problem: ioh.problem.PBO, mu=5, lambda_=20, budget=5000) -> None:
    """
    ES with
    - (mu + lambda) selection
    - Global discrete recombination
    - Self-adaptive mutation 
    """
    n_dim = problem.bounds.lb.shape[0]
    
    # Initialize population randomly
    population = rs.choice(2, (mu, n_dim))
    fitness = np.array([problem(ind) for ind in population])
    
    # Track the best solution found globally
    best_ind = population[np.argmax(fitness)].copy()
    best_fit = np.max(fitness)

    while problem.state.evaluations < budget:
        offspring_pop = []
        offspring_fit = []
        
        # Determine remaining budget
        remaining = budget - problem.state.evaluations
        current_lambda = min(lambda_, remaining)
        
        for _ in range(current_lambda):
            #Recombination: Create a base child from the mu parents
            child = discrete_recombination(population)
            
            #Correlated Mutation Logic:
            # We vary the mutation rate per offspring to explore different scales
            # analogous to step-size control in CMA-ES
            current_mutation_rate = rs.uniform(0.5/n_dim, 2.0/n_dim)
            
            mask = rs.rand(n_dim) < current_mutation_rate
            if not np.any(mask): #Ensure variation
                mask[rs.randint(0, n_dim)] = True
            
            child[mask] = 1 - child[mask]
            
            # Evaluate
            fit = problem(child)
            offspring_pop.append(child)
            offspring_fit.append(fit)
            
            # Update global best tracking
            if fit > best_fit:
                best_fit = fit
                best_ind = child.copy()

        #Selection (mu + lambda)
        # Combine current parents and new offspring
        combined_pop = np.vstack([population, np.array(offspring_pop)])
        combined_fit = np.concatenate([fitness, np.array(offspring_fit)])
        
        # Sort by fitness (descending) and keep top mu
        indices = np.argsort(combined_fit)[::-1]
        population = combined_pop[indices[:mu]]
        fitness = combined_fit[indices[:mu]]

    #Print result at the end of evaluation phase 
    print(f"Algorithm: ES (Adaptive mu+lambda) | Problem: {problem.meta_data.name} "
          f"| Dimension: {n_dim} | Final Best Fitness: {best_fit}")

def create_problem(dimension: int, fid: int) -> typing.Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolutionary_strategy",  # name of your algorithm
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
        student4398270(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        student4398270(F23)
        F23.reset()
    _logger.close()
