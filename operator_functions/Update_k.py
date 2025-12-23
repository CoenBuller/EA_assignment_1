import numpy as np

def update_k(current_iteration, max_iteration, max_k, stagnated_runs):
    if stagnated_runs >= 1200:
        k_new = 3
    else:
        k_new = 2 + (max_k - 2)*(current_iteration/max_iteration)
    return round(k_new)