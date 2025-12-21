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


def initialize(mu: int, problem: object, rs=None):
    """Function is used for the initialization of the population"""
    
    if rs is None:
        rs = RandomState(MT19937(SeedSequence(69))) 

    visited = set() # Set to store all the entire initial population
    pop = []
    duplicates = 0
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
                duplicates += 1


    else:
        while len(pop) != mu:
            individual = rs.choice(2, (problem.bounds.lb.shape[0]), replace=True)
            new, visited = check_visited(individual, visited, problem)
            if new:
                pop.append(individual)
            else:
                duplicates += 1

    print(f"The number of duplicates that got filterd: {duplicates}")
        
    pop = np.array([list(individual) for individual in pop]) # Make sure that all the individuals in the population are stored as lists
    return pop