from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np
from operator_functions.Check_already_visited import check_visited

def hamming_distance(x1, x2):
    return sum(x1 != x2)


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

    offspring = []
    paired_parents = []
    while (len(offspring) < len(pop)):
        mating_idx = rs.choice(pop.shape[0], size=2, p=weights, replace=True)
        mating = pop[mating_idx]
        p1, p2 = mating[0].copy(), mating[1].copy()

        # 2. Uniform crossover operation
        o1, o2 = cross(p1, p2, crossover_probability=p_cross)

        # 3. Implement first stage of crowding algorithm
        possible_pairs = [(p1, o1), (p1, o2), (p2, o1), (p2, o2)]

        first_pair, min_dist = (), np.inf
        for pair in possible_pairs:
            dist = hamming_distance(pair[0], pair[1])
            if dist < min_dist:
                min_dist = dist
                first_pair = pair
        
        possible_pairs = [pair if pair[0] != closest_pair[0] for pair in possible_pairs]
        second_pair = possible_pairs[0]
        if hamming_distance(possible_pairs[1][0], possible_pairs[1][1]) < hamming_distance(second_pair[0], second_pair[1]):
            second_pair = possible_pairs[1]

        offspring.extend([first_pair[1], second_pair[1]])
        paired_parents.extend([first_pair[0], second_pair[1]])


        """Still need to implement that the parents fitness is also stored. This can be used for the selection phase to determine
        if child is better than parent """
    return np.array(offspring), np.array(paired_parents)