from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np


def crossover(pop, pop_f, p_cross, rs=None):
    """
    Function to handle the crossover operation.
    It uses a selection scheme weighted by fitness (softmax)
    to select mating pairs, and then applies uniform crossover.
    """

    if rs is None:
        rs = RandomState(MT19937(SeedSequence(69))) 

    def cross(p1, p2, crossover_probability):
        # Check if crossover should happen at all for this pair
        if rs.rand() < crossover_probability:
            # Apply uniform crossover: iterate through the bitstring length (len(p1))
            for i in range(len(p1)):
                prob = rs.rand()
                # If a random number is less than or equal to 0.5, swap the bits (uniform crossover)
                if prob <= 0.5:
                    p1[i], p2[i] = p2[i], p1[i]
        return p1, p2
    
    # 1. Selection of Mating Pairs (Softmax-Weighted)
    exp_p = np.exp(pop_f - np.max(pop_f)) # Shift by max(pop_f) for numerical stability
    weights = exp_p/(np.sum(exp_p)) # Normalize to sum to 1

    # For crossover we need for each offspring two parents. This means that we need to sample 2 * pop number of parents
    mating_idx = rs.choice(pop.shape[0], size=2*len(pop), p=weights, replace=True)
    mating = pop[mating_idx]

    # 2. Crossover Operation
    offspring = []
    # Iterate through the mating pool, taking two parents at a time
    for i in range(1, len(mating), 2):
        # Apply the uniform crossover function to the pair
        o1, o2 = cross(mating[i-1].copy(), mating[i].copy(), crossover_probability=p_cross)
        offspring.extend([o1, o2])

    return np.array(offspring)