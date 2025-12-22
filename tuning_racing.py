import numpy as np
from sobol_sampling import draw_sobol_samples
from ioh import get_problem, ProblemClass
from ES import student4398270 

def run_race_es_continuous(total_tuning_budget=100000):
    print("--- Tuning Continuous ES (F23 Katsuura) with Successive Halving ---")

    # Search Space (Sobol)
    # Params: [mu, lambda, initial_sigma, adaptation_strength]
    bounds = np.array([
        [5, 50],       # mu: Start small-ish to allow IPOP growth
        [20, 200],     # lambda: Needs to be >> mu
        [0.5, 1.5],    # initial_sigma: Domain is [-5, 5], so step size 1-4 covers well
        [0.1, 0.6]     # adaptation_strength: Around 1/sqrt(10) ~= 0.3
    ])
    
    # Generate 25 candidate configurations
    n_configs = 25
    configs = draw_sobol_samples(*bounds, n_dims=4, n_samples=n_configs)
    
    candidates = []
    for cfg in configs:
        mu_val = int(cfg[0])
        lambda_val = int(cfg[1])
        
        # Ensure lambda is at least 2x mu (basic ES rule)
        if lambda_val < 2 * mu_val:
            lambda_val = 2 * mu_val
            
        candidates.append({
            "mu": mu_val,
            "lambda_": lambda_val,
            "initial_sigma": float(cfg[2]),
            "adaptation_strength": float(cfg[3]),
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
                    mu=candidate["mu"], 
                    lambda_=candidate["lambda_"],
                    budget=budget_per_run,
                    initial_sigma=candidate["initial_sigma"],
                    adaptation_strength=candidate["adaptation_strength"]
                )
                
                # Lower is better
                avg_score += problem.state.current_best.y
                used_budget += budget_per_run
            
            candidate["score"] = avg_score / repetitions
            print(f"Config {i}: mu={candidate['mu']}, lam={candidate['lambda_']}, "
                  f"sig={candidate['initial_sigma']:.2f} -> Score: {candidate['score']:.4f}")

        # Selection (min)
        active_candidates = [c for c in candidates if c["active"]]
        # Sort Ascending (Lower score is better for BBOB)
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