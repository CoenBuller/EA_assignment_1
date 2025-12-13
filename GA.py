from typing import Tuple 
import numpy as np
# Import necessary components from the IOH profiling library
import ioh
import pandas as pd
from ioh import get_problem, logger, ProblemClass
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# Initialize a reproducible random state using a seed sequence
# This is crucial for making your experiments (like tuning) reproducible.
rs = RandomState(MT19937(SeedSequence(69))) 

def crossover(pop, pop_f, p_cross):
    """
    Function to handle the crossover operation.
    It uses a selection scheme weighted by fitness (softmax)
    to select mating pairs, and then applies uniform crossover.
    """

    def cross(p1, p2, crossover_probability):
        # Check if crossover should happen at all for this pair
        if rs.rand() < crossover_probability:
            # Apply uniform crossover: iterate through the bitstring length (len(p1))
            for i in range(len(p1)):
                prob = rs.rand()
                # If a random number is less than or equal to 0.5, swap the bits (uniform crossover)
                if prob <= 0.5:
                    p1[i], p2[i] = p2[i], p1[i]
        return p1, p2
    
    # 1. Selection of Mating Pairs (Softmax-Weighted)
    exp_p = np.exp(pop_f - np.max(pop_f)) # Shift by max(pop_f) for numerical stability
    weights = exp_p/(np.sum(exp_p)) # Normalize to sum to 1

    # For crossover we need for each offspring two parents. This means that we need to sample 2 * pop number of parents
    mating_idx = rs.choice(pop.shape[0], size=2*len(pop), p=weights, replace=True)
    mating = pop[mating_idx]

    # 2. Crossover Operation
    offspring = []
    # Iterate through the mating pool, taking two parents at a time
    for i in range(1, len(mating), 2):
        # Apply the uniform crossover function to the pair
        o1, o2 = cross(mating[i-1].copy(), mating[i].copy(), crossover_probability=p_cross)
        offspring.extend([o1, o2])

    return np.array(offspring)


def mutation(offspring, p_mutate):
    """
    Helper function to apply the mutation (bit-flip) operator.
    """

    p_not_mutate = 1 - p_mutate # Probability the bit will NOT mutate
    
    # Create a mask array: 1 means mutate (bit-flip), 0 means keep (no change)
    # The probability of selecting '1' (mutating) is `p_mutate`.
    mutation_array = rs.choice(2, offspring.shape, replace=True, p=[p_not_mutate, p_mutate]) 
    
    # Apply bit-flip mutation:
    # offspring - mutation_array performs the XOR operation on the bits:
    # 0 - 0 = 0 (No change)
    # 1 - 0 = 1 (No change)
    # 0 - 1 = -1 (Becomes 1 when taking abs) -> 1 (Flip)
    # 1 - 1 = 0 (Flip)
    # np.abs(...) ensures the result is a binary string.
    offspring = np.abs(offspring - mutation_array).astype(np.int64)
    return offspring


def selection(parents, parents_f, offspring, offspring_f, mu_plus_lambda, top_k):
    """
    Selection operator, implementing either (mu + lambda) or (mu, lambda).
    The best 'top_k' individuals are selected to form the next generation.
    """
    
    # (mu + lambda) selection: Combine parents and offspring
    if mu_plus_lambda:
        population, population_f = np.vstack([parents, offspring]), np.hstack([parents_f, offspring_f])
    # (mu, lambda) selection: Only consider the offspring
    else: 
        population, population_f = offspring, offspring_f

    # Sort the population based on fitness (population_f) in descending order (highest fitness is best)
    # np.argsort returns the indices that would sort the array
    best_idx = np.argsort(-population_f)[:top_k]
    
    # Return the top_k individuals and their fitness scores
    return population[best_idx], population_f[best_idx]


def s2631415_studentnumber2_GA(problem: ioh.problem.PBO, mu_plus_lambda=True, mu=10, p_crossover=0.5, mutation_r=0.02, budget=5000) -> tuple[float|int, float|int, float]:
    """
    The main Genetic Algorithm (GA) loop.
    
    Args:
        problem: The IOHprofiler problem instance (F18 or F23).
        mu_plus_lambda (bool): Use (mu + lambda) if True, (mu, lambda) if False.
        mu (int): Population size (mu).
        p_crossover (float): Crossover probability.
        mutation_r (float): Mutation rate (probability of flipping a single bit).
        budget (int): Maximum number of function evaluations (FEs).
    """

    # Initialization: Create the initial population (mu individuals)
    # The individual length is determined by the problem dimension (problem.bounds.lb.shape[0])
    parents = rs.choice(2, (mu, problem.bounds.lb.shape[0]), replace=True)
    
    # Evaluate the initial population
    parents_f = problem(parents) # problem(pop) automatically evaluates all individuals in the batch
    min_f, max_f = min(parents_f), max(parents_f) # Min, and max evaluation values

    # Main Evolutionary Loop
    # Check the number of evaluations using problem.state.evaluations (automatically tracked by IOH)
    while problem.state.evaluations < budget:
        # Update min and max for normalization of the scores

        # 1. Crossover: Generate the offspring population (lambda = mu in this setup)
        offspring = crossover(parents, parents_f, p_cross=p_crossover)
        
        # 2. Mutation: Apply bit-flip mutation to the offspring
        offspring = mutation(offspring, mutation_r)
        
        # 3. Evaluation: Evaluate the new offspring
        offspring_f = np.array([problem(offspring[i]) for i in range(len(offspring))])
        
        # 4. Selection: Select the next generation (parents)
        parents, parents_f = selection(parents, parents_f, offspring, offspring_f, mu_plus_lambda, mu)


        # Debug print: Shows the best fitness near the end of the run
        if problem.state.evaluations > 4990:
            ...
            # print(f"Problem: {problem}, Evaluations: {problem.state.evaluations}, Best: {np.max(parents_f)}")
        
    return min_f, max_f, max(parents_f)


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    """
    Utility function to create an IOHprofiler problem instance and attach a logger.
    """
    # Declaration of the problem instance (fid=18 for LABS, fid=23 for N-Queens)
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    l = logger.Analyzer(
        root="data",  # working directory where the results folder will be created
        folder_name="run",  # folder name for storing the raw performance data
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # Attach the logger to the problem so every evaluation is recorded
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # Example usage: This block is typically for final runs, not tuning.
    
    # F18 (LABS) - Dimension 50
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        min18, max18, maximum18 = s2631415_studentnumber2_GA(F18)
        print(f"\n Standardized increase compared to parents for F18 problem: {abs((maximum18-max18)/(max18))}")
        print(f"Absolute best: {maximum18} | Parents best: {max18} | Parents worst: {min18}")

        F18.reset() # Reset is necessary to start a new independent run with clean state/FE count
    _logger.close() # Close logger to ensure all data is written to the file system

    # F23 (N-Queens) - Dimension 49 (7x7 board)
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        min23, max23, maximum23 = s2631415_studentnumber2_GA(F23)
        print(f"\n Standardized increase compared to parents for F23 problem: {abs((maximum23-max23)/(max23))}")
        print(f"Absolute best: {maximum23} | Parents best: {max23} | Parents worst: {min23}")
        F23.reset()
    _logger.close()

    
# The next step should be to update tuning.py based on the plan we discussed:
# 1. Select only 5 configurations.
# 2. Run each configuration for R=2 runs on both F18 and F23.
# 3. Fix the budget to 5000 FEs in the tuning loop.
# 4. Implement the logic to read the logger files, normalize the scores, and calculate the final tuning score S.