import ioh
from ioh import get_problem, logger, ProblemClass
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

# Initialize Random State
rs = RandomState(MT19937(SeedSequence(69)))
budget = 5000
dimension = 10

def student4398270_student2631415_ES(problem: ioh.problem.BBOB,       
                   budget=5000,
                   initial_sigma=1.44, 
                   return_history=False, # flag for analysis
                   stagnation_limit = 73,
                   sigma_threshold = 0.000132,
                   factor_fail = 0.334

                   ) -> None:
    """
    (1+10) Evolution Strategy with Variance Control (var_ctrl) for BBOB F23.
    """
    
    # --- Configuration ---
    mu_local = 1
    lambda_local = 10
    
    D = problem.meta_data.n_variables
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    
    # Variance Control Parameters
    factor_success = np.exp(1.0 / D)
    factor_fail = np.exp(-factor_fail / D)
    
    # Restart Triggers
    SIGMA_THRESHOLD = sigma_threshold
    STAGNATION_LIMIT = stagnation_limit
    
    # History tracking for local analysis
    history_trace = [] # List of (evals, best_y)

    # --- Main Loop ---
    while problem.state.evaluations < budget:
        
        # Initialization (Start/Restart)
        x_parent = rs.uniform(lb, ub, D)
        sigma = initial_sigma
        
        # Evaluate Parent
        f_parent = problem(x_parent)
        
        # Log initial point
        if return_history:
            history_trace.append((problem.state.evaluations, problem.state.current_best.y))
        
        gens_without_improv = 0
        
        # Inner Generation Loop
        while problem.state.evaluations < budget:
            Z = rs.normal(0, 1, (lambda_local, D))
            
            best_offspring_val = float('inf')
            best_offspring_x = None
            n_better = 0 
            
            for k in range(lambda_local):
                if problem.state.evaluations >= budget: break
                
                # Mutation & Correction
                y_k = x_parent + sigma * Z[k]
                y_k = np.clip(y_k, lb, ub)
                
                # Evaluate
                f_k = problem(y_k)
                
                # Track best in batch
                if f_k < best_offspring_val:
                    best_offspring_val = f_k
                    best_offspring_x = y_k
                
                if f_k < f_parent:
                    n_better += 1

                # Log every evaluation for high-res history
                if return_history:
                    # We log the GLOBAL best so far, not just the current offspring
                    history_trace.append((problem.state.evaluations, problem.state.current_best.y))
            
            if best_offspring_x is None: break
            
            # Adaptation (Variance Control)
            if n_better > 0:
                sigma *= factor_success
            else:
                sigma *= factor_fail
                
            # Selection
            if best_offspring_val <= f_parent:
                if best_offspring_val < f_parent:
                    gens_without_improv = 0
                else:
                    gens_without_improv += 1
                x_parent = best_offspring_x
                f_parent = best_offspring_val
            else:
                gens_without_improv += 1
            
            # Restart Checks
            if sigma < SIGMA_THRESHOLD or gens_without_improv >= STAGNATION_LIMIT:
                break
    print(f"Algorithm: ES (variance control) | Problem: {problem.meta_data.name} "
          f"| Best: {problem.state.current_best.y-problem.optimum.y}")          
    if return_history:
        return history_trace
    
def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run_F23_stats",  # the folder name to which the raw performance data will be stored
        algorithm_name="1+10_EA_var_ctrl",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l

if __name__ == "__main__":
    DIMENSION = 10
    FID = 23
    
    print(f"--- Running (1+10) EA_var_ctrl on F{FID} ({20} runs) ---")
    
    F23, _logger = create_problem(23)
    
    all_run_histories = []
    final_bests = []
    
    for run in range(20):

        history = student4398270_student2631415_ES(F23, budget=5000, return_history=True)
        #Store Data
        all_run_histories.append(history)
        final_bests.append(F23.state.current_best.y)
        
        F23.reset()
        
    _logger.close() 
    print(f"IOH Logs saved to 'data/run_F23_stats")
    