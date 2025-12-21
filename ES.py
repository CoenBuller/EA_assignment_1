import ioh
from ioh import get_problem, logger, ProblemClass
import sys
import numpy as np
import typing
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# Initialize Random State
rs = RandomState(MT19937(SeedSequence(69)))

def two_point_crossover(parents):
    """
    2-Point Crossover: Preserves contiguous blocks of the sequence.
    Expects parents shape (mu, n) but typically uses the first 2 rows 
    if strictly bisexual, or averages across mu for ES
    
    For standard ES recombination, we often pick 2 distinct parents 
    randomly from the mu pool
    """
    mu, n = parents.shape
    # Randomly select two parents from the pool
    idx = rs.choice(mu, 2, replace=False)
    p1, p2 = parents[idx[0]], parents[idx[1]]
    
    # Select two crossover points
    points = np.sort(rs.choice(range(1, n), 2, replace=False))
    c1, c2 = points[0], points[1]
    
    # Create offspring by stitching segments
    offspring = np.concatenate([p1[:c1], p2[c1:c2], p1[c2:]])
    return offspring

def discrete_recombination(parents):
    """
    Uniform recombination (Global): For each bit, randomly pick from any parent.
    """
    mu, n = parents.shape
    offspring = np.zeros(n, dtype=int)
    for i in range(n):
        # Pick a random parent for this specific bit
        offspring[i] = parents[rs.randint(mu), i]
    return offspring

def local_search_best_improvement(problem, individual, current_fitness):
    """
    
    """
    n = len(individual)
    best_ind = individual.copy()
    best_val = current_fitness 
    
    check_order = np.random.permutation(n)
    
    for i in check_order:
        best_ind[i] = 1 - best_ind[i]
        new_val = problem(best_ind)
        
        if new_val > best_val:
            return best_ind, new_val
        else:
            best_ind[i] = 1 - best_ind[i] # Revert
            
    return best_ind, best_val

def student4398270(problem: ioh.problem.PBO, 
                   mu=10, 
                   lambda_=40, 
                   budget=5000,
                   crossover_type="two_point", # Options: 'uniform', 'two_point'
                   initial_rate=0.04112,          # If None, defaults to 1/n
                   adaptation_strength=0.9235     # Controls how drastically sigma changes (tau)
                   ) -> None:
    """
    ES with:
    - (mu + lambda) selection
    - Configurable Recombination (Uniform vs 2-Point)
    - True Self-Adaptive Mutation (Log-normal update)
    """
    n_dim = problem.bounds.lb.shape[0]
    
    #Initialization 
    # Population: (mu, n_dim)
    population = rs.choice(2, (mu, n_dim))
    
    # Mutation Rates (Sigmas): One per individual
    # Default start is 1/n (standard rule of thumb)
    if initial_rate is None:
        start_sigma = 1.0 / n_dim
    else:
        start_sigma = initial_rate
        
    mutation_rates = np.full(mu, start_sigma)
    
    # Evaluate initial population
    fitness = np.array([problem(ind) for ind in population])
    
    # Track global best
    best_ind = population[np.argmax(fitness)].copy()
    best_fit = np.max(fitness)

    # Learning rate (tau) for self-adaptation
    # Standard heuristic: 1 / sqrt(n)
    tau = adaptation_strength / np.sqrt(n_dim)

    # Main Loop 
    while problem.state.evaluations < budget:
        offspring_pop = []
        offspring_fit = []
        offspring_sigmas = []
        
        # Adjust lambda if we are nearing the budget
        remaining = budget - problem.state.evaluations
        current_lambda = min(lambda_, remaining)
        
        for _ in range(current_lambda):

            # Recombination 
            if crossover_type == "two_point":
                child = two_point_crossover(population)
                # Inherit sigma from one of the parents involved (randomly)
                # For simplicity in global pools, we can pick a random parent's sigma
                parent_sigma = mutation_rates[rs.randint(mu)]
            else:
                child = discrete_recombination(population)
                # In global discrete, we usually average sigmas or pick one
                parent_sigma = np.mean(mutation_rates)

            # Self-Adaptive Mutation 
            # 1. Mutate the strategy parameter (sigma) first (Log-Normal)
            # Clip to [1/n, 0.5] to ensure we always flip something but don't randomize completely
            learning_noise = rs.normal(0, 1)
            child_sigma = parent_sigma * np.exp(tau * learning_noise)
            child_sigma = np.clip(child_sigma, 1.0/(2*n_dim), 0.5)
            
            # 2. Mutate the object variables (bits) using the new sigma
            mask = rs.rand(n_dim) < child_sigma
            if not np.any(mask): 
                # Ensure at least one bit flip if the mask is empty
                mask[rs.randint(0, n_dim)] = True
            
            child[mask] = 1 - child[mask]
            
            #  Evaluation 
            fit = problem(child)
            
            offspring_pop.append(child)
            offspring_fit.append(fit)
            offspring_sigmas.append(child_sigma)
            
            if fit > best_fit:
                best_fit = fit
                best_ind = child.copy()

        # Selection (mu + lambda) 
        # Combine parents and offspring
        combined_pop = np.vstack([population, np.array(offspring_pop)])
        combined_fit = np.concatenate([fitness, np.array(offspring_fit)])
        combined_sigmas = np.concatenate([mutation_rates, np.array(offspring_sigmas)])
        
        # Sort desc by fitness
        indices = np.argsort(combined_fit)[::-1]
        
        # Truncate to top mu
        top_indices = indices[:mu]
        population = combined_pop[top_indices]
        fitness = combined_fit[top_indices]
        mutation_rates = combined_sigmas[top_indices]

    print(f"Algorithm: ES (Adaptive) | Problem: {problem.meta_data.name} "
          f"| Best: {best_fit}")

def create_problem(dimension: int, fid: int) -> typing.Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
    l = logger.Analyzer(
        root="data",
        folder_name="run",
        algorithm_name="evolutionary_strategy_adaptive",
        algorithm_info="Adaptive ES with n-point crossover",
    )
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
