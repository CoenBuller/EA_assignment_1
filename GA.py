from typing import Tuple
import os

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence 
import numpy as np

# Import necessary components from the IOH profiling library
import ioh 
from ioh import get_problem, logger, ProblemClass

# Import custom operator functions
from operator_functions.Initializer import initialize
from operator_functions.Crossover import crossover
from operator_functions.Mutate import mutation

from operator_functions.Selector import selection
from operator_functions.Update_k import update_k
from operator_functions.Evaluate import Evaluate
from operator_functions.Local_search import local_search



def s2631415_studentnumber2_GA(problem: ioh.problem.PBO, mu_plus_lambda=True, mu=20, p_crossover=0.5, mutation_r=0.02, k_max=4, budget=5000, local_freq=5, seed=69) -> tuple[float|int, float|int, float]:
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
    rs = RandomState(MT19937(SeedSequence(seed)))
    evaluater = Evaluate(problem, budget)

    # Initialization: Create the initial population (mu individuals)
    # The individual length is determined by the problem dimension (problem.bounds.lb.shape[0])
    parents, visited = initialize(mu, problem, rs=rs)
    
    # Evaluate the initial population
    parents_f = evaluater.eval(parents) # problem(pop) automatically evaluates all individuals in the batch
    min_f, max_f = min(parents_f), max(parents_f) # Min, and max evaluation values

    stagnated_runs = 0
    while evaluater.eval_count < budget:        
        # 1. Crossover: Generate the offspring population (lambda = mu in this setup)
        k = update_k(current_iteration=evaluater.iterations, 
                    max_iteration=evaluater.budget, 
                    max_k=k_max, 
                    stagnated_runs=stagnated_runs)

        offspring = crossover(parents, parents_f, p_crossover, k, rs=rs)
        
        # 2. Mutation: Apply bit-flip mutation to the offspring
        offspring = mutation(offspring, mutation_r, rs=rs)
        
        # 3. Evaluation: Evaluate the new offspring
        offspring_f = evaluater.eval(offspring)
        
        current_best = max(parents_f)

        # 4. Selection: Select the next generation (parents)
        parents, parents_f = selection(parents, parents_f, offspring, offspring_f, mu)

        if evaluater.iterations % local_freq == 0:
            local_search_individual, local_search_fitness_score = local_search(parents[0], parents_f[0], evaluater)
            if local_search_individual is not None:
                parents[0], parents_f[0] = local_search_individual, local_search_fitness_score

        new_best_f = max(parents_f)
        
        if new_best_f <= current_best:
            stagnated_runs += mu
        else:
            stagnated_runs = 0
        
        
    return min_f, max_f, max(parents_f)


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    """
    Utility function to create an IOHprofiler problem instance and attach a logger.
    """
    # Declaration of the problem instance (fid=18 for LABS, fid=23 for N-Queens)
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO) # type: ignore

    # Create default logger compatible with IOHanalyzer
    l = logger.Analyzer(
        root="data",  # working directory where the results folder will be created # type:ignore
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
    f18_performance = []
    for run in range(10): 
        seed = run
        min18, max18, maximum18 = s2631415_studentnumber2_GA(F18, mu=64, p_crossover=0.22707223864798573, mutation_r=0.049023001749565254, seed=seed)
        print(f"\n Standardized increase compared to parents for F18 problem: {abs((maximum18-max18)/(max18))}")
        print(f"Absolute best: {maximum18} | Parents best: {max18} | Parents worst: {min18}")
        f18_performance.append(maximum18)

        F18.reset() # Reset is necessary to start a new independent run with clean state/FE count
    _logger.close() # Close logger to ensure all data is written to the file system # type: ignore

    # F23 (N-Queens) - Dimension 49 (7x7 board)
    F23, _logger = create_problem(dimension=49, fid=23)
    f23_performance = []
    for run in range(10): 
        seed=run
        min23, max23, maximum23 = s2631415_studentnumber2_GA(F23, mu=64, p_crossover=0.22707223864798573, mutation_r=0.049023001749565254, seed=seed)
        f23_performance.append(maximum23)
        print(f"\n Standardized increase compared to parents for F23 problem: {abs((maximum23-max23)/(max23))}")
        print(f"Absolute best: {maximum23} | Parents best: {max23} | Parents worst: {min23}")

        print(F23)

        F23.reset()
    _logger.close() # type: ignore

    print(f"\nPerformance of GA on F18 \n      Mean: {np.mean(f18_performance)} | Std: {np.std(f18_performance)}")
    print(f"Performance of GA on F23 \n      Mean: {np.mean(f23_performance)} | Std: {np.std(f23_performance)}")


    
