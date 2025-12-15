from scipy.stats import qmc
import numpy as np

def draw_sobol_samples(*bounds_list, n_dims, n_samples=2**10, scramble=True, seed=None):
    """
    
    Sobol sampler with variable bounds list. 
    

    Parameters:

    n_dims: dimensionality of problem
    *bounds_list: iterable containing the lower and upper bound to sample between
    n_samples: number of samples to draw. Works best if it is a multiple of 2
    scramble: wether to scramble the sequence. Reduces correlation
    seed: random seed
    '"""

    bounds = np.array(bounds_list) # Convert to array for consistency

    sampler = qmc.Sobol(d=n_dims, scramble=scramble, seed=seed)

    if (n_samples & (n_samples - 1)) == 0: 
        m = int(np.log2(n_samples))
        samples = sampler.random_base2(m)
    else:
        samples = sampler.random(n_samples)

    scaled_samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1]) # Apply bounds

    return list(scaled_samples)