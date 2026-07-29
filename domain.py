


import numpy as np
from scipy.stats import qmc


# define uniform samples on the domain
# specify the dimension of the domain and domain range
def domain_samples(num_samples: int, dim: int, domain_range: np.ndarray) -> np.ndarray:
   
    # num_samples is the number of samples to generate
    # dim is the dimension of the domain
    # domain_range is a 2d array of shape (2, dim), where the first row is the lower bound and the second row is the upper bound of the domain 

    sampler = qmc.LatinHypercube(d = dim)
    samples = sampler.random(n = num_samples)
    # scale the samples to the domain range
    scaled_samples = qmc.scale(samples, l_bounds = domain_range[0], u_bounds = domain_range[1])
   
    return scaled_samples

# # decompose domain samples into equivalent classes by each forward model.
def equivalent_classes(domain_samples: np.ndarray, 
                       forward_model: callable, 
                       discretization: tuple ) -> np.ndarray:
    
    # domain_samples: a 2d array of shape (num_samples,dim), 
    # forward_model: a callable function that takes in domain samples and returns 1D data samples
    # a and b are coefficients for the forward model 

    data_samples = forward_model(domain_samples)
    index = (data_samples - discretization[1])/ discretization[2] 
    inbound = (index >= 0) & (index <= len(discretization[0]))
    index[inbound] = np.floor(index[inbound]).astype(int) 
    index[~inbound] = -1 # assign -1 to samples that are out of bounds of the discretized data range
    
    return index










