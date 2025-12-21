from typing import Tuple
import numpy as np
from tqdm import tqdm
from GA import s2631415_studentnumber2_GA, create_problem, initialize


# Tuning settings
TOTAL_TUNING_BUDGET = 100_000
BUDGET_PER_RUN = 5_000     # per problem
SEED = 6

np.random.seed(SEED)

# Hyperparameter ranges
MU_RANGE = (20, 250)
MUT_RANGE = (0.02, 0.1)
CROSS_RANGE = (0.1, 0.5)

# Each config costs 2 * BUDGET_PER_RUN
COST_PER_CONFIG = 2 * BUDGET_PER_RUN
N_CONFIGS = TOTAL_TUNING_BUDGET // COST_PER_CONFIG


# Start tuning
def sample_config() -> Tuple[int, float, float]:
    """Sample one random hyperparameter configuration."""
    p_mut = np.random.uniform(*MUT_RANGE)
    p_cross = np.random.uniform(*CROSS_RANGE)
    mu = np.random.uniform(*MU_RANGE)
    return int(mu), p_mut, p_cross


def tune_hyperparameters():
    best_score = (-np.inf, -np.inf)
    best_params = (None, None, None)
    print(f"Testing {N_CONFIGS} configurations")
    for _ in tqdm(range(2 * N_CONFIGS)):
        mu, p_mut, p_cross = sample_config()

        # --- F18 ---
        F18, _ = create_problem(dimension=50, fid=18)
        init_pop18 = initialize(mu, F18)

        _, init_max18, best18 = s2631415_studentnumber2_GA(
            problem=F18,
            mu=mu,
            p_crossover=p_cross,
            mutation_r=p_mut,
            budget=BUDGET_PER_RUN,
            initial_pop=init_pop18
        )

        # # --- F23 ---
        # F23, _ = create_problem(dimension=49, fid=23)
        # init_pop23 = initialize(mu, F23)

        # _, init_max23, best23 = s2631415_studentnumber2_GA(
        #     problem=F23,
        #     mu=mu,
        #     p_crossover=p_cross,
        #     mutation_r=p_mut,
        #     budget=BUDGET_PER_RUN,
        #     initial_pop=init_pop23
        # )

        # Simple improvement score
        score18 = best18
        score23 = 0
        total_score = score18 + score23

        if total_score > sum(best_score):
            best_score = (score18, score23)
            best_params = (mu, p_mut, p_cross)
        

    return best_params, best_score


if __name__ == "__main__":
    best_params, best_score = tune_hyperparameters()
    score18, score23 = best_score[0], best_score[1]
    mu, mutation_rate, crossover_rate = best_params[0], best_params[1], best_params[2]

    print(f"Best score for F18: {best_score[0]}")
    print(f"Best score for F23: {best_score[1]}")
    print("Best configuration found:")
    print("mu =", mu)
    print("mutation rate =", mutation_rate)
    print("crossover rate =", crossover_rate)
