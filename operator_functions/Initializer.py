import ioh # type: ignore
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

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

    if isinstance(problem, ioh.iohcpp.problem.NQueens): # Initialization for F23 
        """Ïn each row and each column we need at most one queen. 
        We can leverage this during the initialization. Now the algorithm only needs to sort out the diagonal problem"""
        pop = set() # Set to store all the individuals in
        queen_pos = np.arange(0, 7) 
        for _ in range(mu):
            while len(pop) != mu: # Keep making new permutations untill the set size is the same as mu
                individual = rs.permutation(queen_pos)
                individual = decode_queens(individual)
                assert(len(individual) == 49)
                pop.add(tuple(individual.astype(int)))
        pop = np.array([list(individual) for individual in pop])
        return pop

    return rs.choice(2, (mu, problem.bounds.lb.shape[0]), replace=True) # initialization for F18 # type: ignore