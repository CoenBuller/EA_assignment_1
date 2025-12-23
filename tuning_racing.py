import numpy as np
from sobol_sampling import draw_sobol_samples
from ioh import get_problem, ProblemClass
from ES import student4398270 

def run_race_es_continuous(total_tuning_budget=100000):
    print("--- Tuning Continuous ES (F23 Katsuura) with Successive Halving ---")

    # Search Space (Sobol)
    # Params
    bounds = np.array([
    [0.5,4.0], #Initial sigma
    [10, 100],      # Stagnation Limit
    [1e-8, 1e-3],   # Sigma Threshold (More conservative range)
    [0.05, 0.4]     # Factor Fail Coefficient (Small = Slow decay, Large = Fast decay)
])
    
    # Generate 25 candidate configurations
    n_configs = 25
    configs = draw_sobol_samples(*bounds, n_dims=4, n_samples=n_configs)
    
    candidates = []
    for cfg in configs:
        initial_sigma = float(cfg[0])
        stagnation_limit = int(cfg[1])
        sigma_threshold = float(cfg[2])
        factor_fail = float(cfg[3])
      
            
        candidates.append({
            "initial_sigma": initial_sigma,
            "stagnation_limit": stagnation_limit,
            "sigma_threshold": sigma_threshold,
            "factor_fail": factor_fail,
            "active": True,
            "score": float('inf') # Initialize with inf for Minimization
        })

    # Racing Schedule
    rounds = [
        {"evals": 1500, "reps": 2, "keep": 12},  
        {"evals": 3500, "reps": 3, "keep": 5},  
        {"evals": 5000, "reps": 5, "keep": 1}    # (Final budget limit)
    ]

    used_budget = 0

    for r_idx, round_cfg in enumerate(rounds):
        budget_per_run = round_cfg["evals"]
        repetitions = round_cfg["reps"]
        n_keep = round_cfg["keep"]
        
        print(f"\n--- Round {r_idx+1}: {budget_per_run} evals x {repetitions} reps ---")
        
        for i, candidate in enumerate(candidates):
            if not candidate["active"]:
                continue
            
            avg_score = 0
            
            for _ in range(repetitions):

                problem = get_problem(23, dimension=10, instance=1, problem_class=ProblemClass.BBOB)
                
                student4398270(
                    problem, 
                    initial_sigma=candidate["initial_sigma"], 
                    stagnation_limit=candidate["stagnation_limit"],
                    budget=budget_per_run,
                    sigma_threshold=candidate["sigma_threshold"],
                    factor_fail=candidate["factor_fail"]
                )
                
                # Lower is better
                avg_score += problem.state.current_best.y
                used_budget += budget_per_run
            
            candidate["score"] = avg_score / repetitions
            print(f"Config {i}: initial_sigma={candidate['initial_sigma']}, stagnation_limit={candidate['stagnation_limit']}, "
                  f"sigma_threshold={candidate['sigma_threshold']:.2f}, factor_fail={candidate['factor_fail']},-> Score: {candidate['score']:.4f}")

        # Selection (min)
        active_candidates = [c for c in candidates if c["active"]]
        # Sort Ascending (Lower score is better)
        active_candidates.sort(key=lambda x: x["score"]) 
        
        print(f"Round {r_idx+1} Best Score: {active_candidates[0]['score']}")
        
        # Kill the worst
        for i in range(len(active_candidates)):
            if i >= n_keep:
                active_candidates[i]["active"] = False
        
        if used_budget > total_tuning_budget:
            print("Warning: Tuning budget exhausted!")
            break

    winner = [c for c in candidates if c["active"]][0]
    print(f"\nWinner found after {used_budget} evaluations:")
    print(winner)
    return winner

if __name__ == "__main__":
    best_params = run_race_es_continuous()