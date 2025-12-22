import numpy as np
import ioh # type: ignore


def convert_to_bipolar(individual):
    bi_individual = 1 - 2 * individual
    return tuple(bi_individual.astype(np.int32))

def check_visited(individual, visited, problem):
    """
    This function will check if two individuals are identical or not based on symmetries in the problem. For it to work properly, we assume that the population is in binary representation
    """

    #------------------------- N-queens ------------------------------
    if isinstance(problem, ioh.iohcpp.problem.NQueens):
        '''To make this work properly, we must enforce all the arrays have the same shape and dtype. We will store the arrays in a set, which requires us to convert it to bytes. 
        If the shape and dtype are not the same, the arrays will produce different outputs when converting it to bytes.  '''

        board_dim = int(np.sqrt(len(individual))) # One dimension of the chess board
        board = np.asarray(individual, dtype=np.int32).reshape((board_dim, board_dim)) # Transform the array to the chess board dimension

        variants = [
                board,
                np.rot90(board, 1),
                np.rot90(board, 2),
                np.rot90(board, 3),
                np.flip(board, 0),
                np.flip(board, 1),
                board.T,
                np.fliplr(board).T,
            ]

        key = min(v.tobytes() for v in variants)

        # If none of the transformed board versions are in the visited set than we can savely add the chess board to the visited set
        if key not in visited:
            visited.add(key)
            return True, visited
        
    #------------------------- LABS ------------------------------
    else: 
        bi_individual = convert_to_bipolar(individual)
        reversed_individual = bi_individual[::-1]
        negative_individual = tuple(-x for x in bi_individual)
        negative_reversed_individual = tuple(-x for x in reversed_individual)   

        # Since a = -a = reversed_a = -reversed_a, we will only add the individual to the population if none of these versions are in the population
        if (bi_individual not in visited) and (negative_individual not in visited) and (reversed_individual not in visited) and (negative_reversed_individual not in visited):
            visited.add(bi_individual)
            return True, visited

    return False, visited

