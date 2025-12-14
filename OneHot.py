import numpy as np

def encode_queens(OneHot):
    '''
    This function is able to encode the bitstring that is used in the N-Queens problem to its special phenotype representation, which is the inverse of 
    a one-hot encoding

    params:
        **OneHot:** the bitstring representation of state

    returns:
        Inverse of one-hot encoding of bitstring
    '''

    return None

def decode_queens(arr):
    '''
    Docstring for decode_queens
    
    :param arr: array where the index represents the row, and the value at that index the column where the queen stands


    returns:
        One-hot encoding
    '''

    one_hot = np.zeros(len(arr)**2)
    for i in range(len(arr)):
        idx = arr[i]
        one_hot[i*7 + idx] = 1

    return one_hot.astype(int)