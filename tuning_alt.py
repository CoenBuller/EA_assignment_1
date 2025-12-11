import numpy as np
import typing
from sobol_sampling import draw_sobol_samples
from ioh import get_problem, logger, ProblemClass
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/

from GA import s2631415_studentnumber2_GA, create_problem as create_problem_ga
from ES import student4398270, create_problem as create_problem_es
from log import IPL


FINAL_BUDGET = 10000  # target budget 
TUNING_ANCHORS = [500, 1000, 1500, 2000, 2500] # various anchor points
REPETITIONS = 3 # avg over a few runs to reduce noise

def evaluate_configuration(algorithm_func, params, problem_fid, dimension):
    """
    Runs the algorithm for increasing budgets and uses IPL to predict final score
    """
    results = []
    
    for budget in TUNING_ANCHORS:
        avg_fitness = 0
        for r in range(REPETITIONS):
            problem = get_problem(problem_fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
            algorithm_func(problem, **params, budget=budget)
            avg_fitness += problem.state.current_best.y
        
        avg_fitness /= REPETITIONS
        results.append((budget, avg_fitness))
    
    # fit the learning curve
    fit_output = IPL.fit_inverse_power(results)
    
    # Unpack the tuple (popt, pcov) from log.py
    if isinstance(fit_output, tuple):
        popt = fit_output[0]
    else:
        popt = fit_output 
    
    # Predict performance
    predicted_score = IPL.inverse_power_law(FINAL_BUDGET, *popt)
    
    return float(predicted_score), results

def tune_ga():
    print("--- Tuning Genetic Algorithm  ---")
    #Define Bounds
    # mu: [5, 100], mutation: [0.001, 0.1], crossover: [0.1, 0.9], mu+lambda: [0, 1] (treated as boolean)
    bounds = np.array([[5, 100], [0.001, 0.1], [0.1, 0.9], [0, 1]])
    
    #Draw Samples
    n_configs = 16 
    configs = draw_sobol_samples(*bounds, n_dims=4, n_samples=n_configs)
    
    best_config = None
    best_pred_score = -float('inf')

    #Evaluate
    for i, cfg in enumerate(configs):
        # Decode parameters
        mu = int(cfg[0])
        mut_r = float(cfg[1])
        cross_r = float(cfg[2])
        mu_plus_lambda = bool(cfg[3] > 0.5)
        
        params = {
            "mu": mu,
            "mutation_r": mut_r, 
            "p_crossover": cross_r,
            "mu_plus_lambda": mu_plus_lambda
        }
        
        # Evaluate on the 2 problems
        pred_score,curve = evaluate_configuration(s2631415_studentnumber2_GA, params, problem_fid=18, dimension=50)
        pred_score2,curve2 = evaluate_configuration(s2631415_studentnumber2_GA, params,problem_fid=23, dimension = 49)
        pred_score += pred_score2
        print(f"Config {i+1}/{n_configs}: {params} -> Pred Score: {pred_score:.2f}")
        print(f"Config {i+1}/{n_configs}: {params} -> Results LABS: {max(curve, key = lambda x: x [1])[1]}")
        print(f"Config {i+1}/{n_configs}: {params} -> Results NQUEENS: {max(curve2, key = lambda x: x [1])[1]}")
        if pred_score > best_pred_score:
            best_pred_score = pred_score
            best_config = params

    print(f"Best GA Config: {best_config}")
    print(f"Best score: {best_pred_score} ")
    return best_config

def tune_es():
    print("\n--- Tuning Evolutionary Strategy  ---")
    #Define Bounds
    # mu: [1, 50], lambda: [1, 50]
    bounds = np.array([[1, 50], [1, 50]])
    
    #Draw Samples
    n_configs = 16
    configs = draw_sobol_samples(*bounds, n_dims=2, n_samples=n_configs)
    
    best_config = None
    best_pred_score = -float('inf')

    #Evaluate
    for i, cfg in enumerate(configs):
        mu = int(cfg[0])
        lambda_ = int(cfg[1])
        
        # lambda must be at least 1, mu at least 1
        mu = max(1, mu)
        lambda_ = max(1, lambda_)
        
        params = {
            "mu": mu,
            "lambda_": lambda_
        }
        
        # Evaluate 
        pred_score, curve = evaluate_configuration(student4398270, params, problem_fid=18, dimension=50)
        pred_score2, curve2 = evaluate_configuration(student4398270, params,problem_fid=23, dimension = 49)
        pred_score +=pred_score2
        print(f"Config {i+1}/{n_configs}: {params} -> Pred Score: {pred_score:.2f}")
        print(f"Config {i+1}/{n_configs}: {params} -> Results LABS: {max(curve, key = lambda x: x [1])[1]}")
        print(f"Config {i+1}/{n_configs}: {params} -> Results NQUEENS: {max(curve2, key = lambda x: x [1])[1]}")
        
        if pred_score > best_pred_score:
            best_pred_score = pred_score
            best_config = params

    print(f"Best ES Config: {best_config}")
    print(f"Best score: {best_pred_score} ")
    return best_config

if __name__ == "__main__":
    best_ga = tune_ga()
    best_es = tune_es()
    
    print("\nFinal Results:")
    print("GA Best:", best_ga)
    print("ES Best:", best_es)