import numpy as np
from operator_functions.Evaluate import Evaluate

def local_search(individual, individual_f, evaluater):
    for i in range(len(individual)):
        copy_individual = individual.copy()
        copy_individual[i] = abs(copy_individual[i] - 1) # flip one bit
        fitness_value = evaluater.eval([copy_individual])
        if fitness_value > individual_f: # If the individual with one changed bit is better we use that individual
            return copy_individual, fitness_value 
        else:
            return None, None

