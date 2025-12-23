import ioh # type: ignore
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from operator_functions.Check_already_visited import check_visited

def decode_queens(arr):
    """
    Docstring for decode_queens
    
    :param arr: board encoding 
    """
    one_hot = np.zeros(len(arr)**2)
    for i in range(len(arr)):
        idx = arr[i]
        one_hot[i*7 + idx] = 1

    return one_hot.astype(int)


def expand_LABS_bits(bitstring):
    assert(len(bitstring) == 25)
    first_half = np.array(bitstring)
    second_half = np.array([abs(bit - (1+i)%2) for i, bit in enumerate(first_half)])
    full_sequence = np.concatenate([first_half, second_half])
    return full_sequence

def initialize(mu: int, problem: object, rs=None):
    """Function is used for the initialization of the population"""
    
    if rs is None:
        rs = RandomState(MT19937(SeedSequence(69))) 

    visited = set() # Set to store all the entire initial population
    pop = []
    if isinstance(problem, ioh.iohcpp.problem.NQueens): # Initialization for F23 
        """In each row and each column we need at most one queen. 
        We can leverage this during the initialization. Now the algorithm only needs to sort out the diagonal problem"""

        queen_pos = np.arange(0, 7) 
        while len(pop) != mu: # Keep making new permutations untill the set size is the same as mu
            individual = rs.permutation(queen_pos)
            individual = decode_queens(individual)
            assert(len(individual) == 49)
            new, visited = check_visited(individual, visited, problem)
            if new:
                pop.append(individual)

    else:
        while len(pop) != mu:
            individual = rs.choice(2, int((problem.bounds.lb.shape[0])), replace=True)
            new, visited = check_visited(individual, visited, problem)
            if new:
                pop.append(individual)

        
    pop = np.array([list(individual) for individual in pop]) # Make sure that all the individuals in the population are stored as lists
    return pop, visited