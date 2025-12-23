from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np
from operator_functions.Check_already_visited import check_visited



def crossover(pop, pop_f, p_cross, k=8, rs=None):
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

    offspring = []
    while (len(offspring) < len(pop)):
        mating_idx = rs.choice(pop.shape[0], size=2*k, replace=True)
        mating = pop[mating_idx]
        fitness_values = np.array(pop_f)[mating_idx]

        p1_idx = np.argmax(fitness_values[:k])
        p2_idx = np.argmax(fitness_values[k:])

        p1, p2 = mating[p1_idx].copy(), mating[p2_idx].copy()


        # 2. Uniform crossover operation
        o1, o2 = cross(p1, p2, crossover_probability=p_cross)
        offspring.extend([o1, o2])

    offspring = np.array(offspring)
    return offspring