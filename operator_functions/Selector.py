import numpy as np

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
