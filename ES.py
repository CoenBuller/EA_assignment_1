import ioh
from ioh import get_problem, logger, ProblemClass
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import matplotlib.pyplot as plt

# Initialize Random State
rs = RandomState(MT19937(SeedSequence(69)))

def student4398270(problem: ioh.problem.BBOB, 
                   mu=1,            
                   lambda_=10,      
                   budget=5000,
                   initial_sigma=3.0, 
                   return_history=False # flag for analysis
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
    factor_fail = np.exp(-0.1 / D)
    
    # Restart Triggers
    SIGMA_THRESHOLD = 1e-12
    STAGNATION_LIMIT = 20
    
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
          f"| Best: {problem.state.current_best.y}")          
    if return_history:
        return history_trace

# --- Analysis & Plotting Helper Functions ---

def calculate_ert(run_histories, targets):
    """
    Calculates Expected Running Time (ERT) for a list of targets.
    ERT = #Evals_Success + #Evals_Fail / Success_Rate
    """
    erts = []
    success_rates = []
    
    for target in targets:
        successes = 0
        total_evals = 0
        successful_evals = 0
        
        for run in run_histories:
            # Check if this run ever hit the target
            # run is list of (evals, y)
            hit = False
            evals_to_hit = run[-1][0] # Default to max evals
            
            for evals, y in run:
                if y <= target:
                    hit = True
                    evals_to_hit = evals
                    break
            
            if hit:
                successes += 1
                successful_evals += evals_to_hit
            else:
                total_evals += run[-1][0] # Add max budget
        
        if successes > 0:
            # ERT Formula: Total evaluations of successful runs + Total evals of failed runs / number of successes
            # ERT = Sum(evals) / successes
            
            # We need Sum of ALL evals (both successful and failed)
            # If a run succeeded, we count evals to target
            # If it failed, we count max evals
            all_evals_sum = successful_evals + (len(run_histories) - successes) * 5000
            ert = all_evals_sum / successes
            erts.append(ert)
            success_rates.append(successes / len(run_histories))
        else:
            erts.append(np.inf)
            success_rates.append(0.0)
            
    return np.array(erts), np.array(success_rates)

def analyze_and_plot(run_data, best_found_values):
    """
    Generates plots and prints stats.
    """
    print("\n--- Empirical Analysis ---")
    
    # 1. Average Best Fitness
    avg_best = np.mean(best_found_values)
    std_best = np.std(best_found_values)
    print(f"Average Best Fitness (20 runs): {avg_best:.4e} (std: {std_best:.2e})")
    
    # 2. AUC (Area Under Curve of ECDF)
    #  We calculate ECDF of the FINAL values
    sorted_bests = np.sort(best_found_values)
    y_axis = np.arange(1, len(sorted_bests) + 1) / len(sorted_bests)
    
    # AUC score usually requires a range of targets. 
    # Here we report the area under this specific ECDF curve
    auc = np.trapz(y_axis, sorted_bests)
    print(f"AUC (approx integration of final values): {auc:.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Convergence Histories (Raw)
    for run in run_data:
        evals, ys = zip(*run)
        axes[0].plot(evals, np.log10(ys), alpha=0.3, color='blue')
    axes[0].set_title("Convergence (All Runs)")
    axes[0].set_xlabel("Evaluations")
    axes[0].set_ylabel("log10(Best Fitness)")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: ECDF (Empirical Cumulative Distribution) of Final Solution
    axes[1].step(sorted_bests, y_axis, where='post', label='Algorithm')
    axes[1].set_title("ECDF (Final Fitness)")
    axes[1].set_xlabel("Best Fitness Reached")
    axes[1].set_ylabel("Proportion of Runs")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    # Plot 3: ERT (Expected Running Time)
    # Define targets from best reached to worst reached
    min_val = np.min(best_found_values)
    max_val = np.max(best_found_values)
    if min_val <= 0: min_val = 1e-8 # Safety for log
    
    targets = np.logspace(np.log10(min_val), np.log10(max_val), 20)
    erts, success_rates = calculate_ert(run_data, targets)
    
    axes[2].plot(targets, erts, 'o-', color='red', label='ERT')
    axes[2].set_title("ERT vs Target Fitness")
    axes[2].set_xlabel("Target Fitness f(x)")
    axes[2].set_ylabel("ERT (Evaluations)")
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].invert_xaxis() # Better targets (lower) on the right usually, or left 
                           # Convention: Lower is better 
                           # If we invert, right is better (lower fitness)
    
    plt.tight_layout()
    plt.show()
    print("Plots generated. Close window to finish.")

if __name__ == "__main__":
    DIMENSION = 10
    FID = 23
    RUNS = 20
    
    print(f"--- Running (1+10) EA_var_ctrl on F{FID} ({RUNS} runs) ---")
    
    # This creates the standard .dat / .json files
    problem_template = get_problem(FID, dimension=DIMENSION, instance=1, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root="data",
        folder_name="run_F23_stats",
        algorithm_name="1+10_EA_var_ctrl",
        algorithm_info="Adaptive ES with Variance Control"
    )
    
    all_run_histories = []
    final_bests = []
    
    for r in range(RUNS):
        # We must re-create or reset the problem and attach logger every time
        # to ensure clean logging
        p = get_problem(FID, dimension=DIMENSION, instance=1, problem_class=ProblemClass.BBOB)
        p.attach_logger(l)
        
        # Run Algorithm and capture internal history
        history = student4398270(p, budget=5000, return_history=True)
        
        # Store Data
        all_run_histories.append(history)
        final_bests.append(p.state.current_best.y)
        
        p.reset()
        
    l.close() # Save IOH files
    print(f"IOH Logs saved to 'data/run_F23_stats")
    
    # Run Local Analysis
    analyze_and_plot(all_run_histories, final_bests)