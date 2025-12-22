import numpy as np
from sobol_sampling import draw_sobol_samples
from ioh import get_problem, ProblemClass
from ES import student4398270 

def run_race_es(total_tuning_budget=100000):
    print("--- Tuning ES with Successive Halving (Racing) ---")

    # 1. Define Search Space (Sobol)
    # Params: [mu (5-15), lambda (20-60), stagnation_limit (15-50), kick_strength (0.05-0.5)]
    # Note: We fix crossover to "two_point" based on your findings.
    bounds = np.array([
        [5, 50],       # mu
        [10, 100],      # lambda
        [0, 100],      # stagnation_limit
        [0.05, 1.0]    # kick_strength (Soft Restart intensity)
    ])
    
    # Initial Pool: 25 configurations
    n_configs = 25
    configs = draw_sobol_samples(*bounds, n_dims=4, n_samples=n_configs)
    
    # Convert Sobol samples to dictionary list
    candidates = []
    for cfg in configs:
        candidates.append({
            "mu": int(cfg[0]),
            "lambda_": int(cfg[1]),
            "stagnation_limit": int(cfg[2]),
            "kick_strength": float(cfg[3]),
            "active": True,
            "score": 0.0
        })

    # 2. Racing Schedule
    # We gradually increase the budget per run while decreasing the # of candidates.
    # Total Cost Estimate:
    # Round 1: 25 configs * 1 runs * 1000 evals = 25,000
    # Round 2: 12 configs * 2 runs * 2000 evals = 48,000
    # Round 3: 5 configs  * 3 runs * 3000 evals = 45,000 (Overshoot prevention logic needed)
    
    rounds = [
        {"evals": 1000, "reps": 1, "keep": 12},
        {"evals": 2500, "reps": 2, "keep": 5},
        {"evals": 5000, "reps": 3, "keep": 1} # Winner
    ]

    used_budget = 0

    for r_idx, round_cfg in enumerate(rounds):
        budget_per_run = round_cfg["evals"]
        repetitions = round_cfg["reps"]
        n_keep = round_cfg["keep"]
        
        print(f"\n--- Round {r_idx+1}: Running {budget_per_run} evals x {repetitions} reps ---")
        
        # Iterate over active candidates
        for i, candidate in enumerate(candidates):
            if not candidate["active"]:
                continue
            
            avg_score = 0
            # To be robust, we sum scores across 2 problems (LABS + NQueens)
            # or just focus on LABS since it's the harder one.
            # Let's use LABS (F18) as the primary filter.
            
            for _ in range(repetitions):
                # Setup Problem
                problem = get_problem(18, dimension=50, instance=1, problem_class=ProblemClass.PBO)
                
                # Run Algorithm (Ensure your ES accepts these new params!)
                student4398270(
                    problem, 
                    mu=candidate["mu"], 
                    lambda_=candidate["lambda_"],
                    budget=budget_per_run,
                    crossover_type="two_point", 
                    stagnation_limit=candidate["stagnation_limit"], 
                    kick_strength=candidate["kick_strength"]        
                )
                
                avg_score += problem.state.current_best.y
                used_budget += budget_per_run
            
            candidate["score"] = avg_score / repetitions
            print(f"Config {i}: {candidate} -> Score: {candidate['score']:.4f}")

        # Selection: Sort by score and kill the weak
        # Filter active candidates
        active_candidates = [c for c in candidates if c["active"]]
        active_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"Round {r_idx+1} Best: {active_candidates[0]['score']}")
        
        # Mark losers as inactive
        for i in range(len(active_candidates)):
            if i >= n_keep:
                active_candidates[i]["active"] = False
        
        if used_budget > total_tuning_budget:
            print("Warning: Tuning budget exhausted!")
            break

    # 3. Output Winner
    winner = [c for c in candidates if c["active"]][0]
    print(f"\nWinner found after {used_budget} evaluations:")
    print(winner)
    return winner

if __name__ == "__main__":
    best_params = run_race_es()