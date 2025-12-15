from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np

def mutation(offspring, p_mutate, rs=None):
    """
    Helper function to apply the mutation (bit-flip) operator.
    """

    if rs is None:
        rs = RandomState(MT19937(SeedSequence(69))) 


    p_not_mutate = 1 - p_mutate # Probability the bit will NOT mutate
    
    # Create a mask array: 1 means mutate (bit-flip), 0 means keep (no change)
    # The probability of selecting '1' (mutating) is `p_mutate`.
    mutation_array = rs.choice(2, offspring.shape, replace=True, p=[p_not_mutate, p_mutate]) 
    
    # Apply bit-flip mutation:
    # offspring - mutation_array performs the XOR operation on the bits:
    # 0 - 0 = 0 (No change)
    # 1 - 0 = 1 (No change)
    # 0 - 1 = -1 (Becomes 1 when taking abs) -> 1 (Flip)
    # 1 - 1 = 0 (Flip)
    # np.abs(...) ensures the result is a binary string.
    offspring = np.abs(offspring - mutation_array).astype(np.int64)
    return offspring