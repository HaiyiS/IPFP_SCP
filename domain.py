import numpy as np
from scipy.stats import qmc

def domain_samples(num_samples: int, dim: int, domain_range: np.ndarray) -> np.ndarray:
    sampler = qmc.LatinHypercube(d = dim)
    samples = sampler.random(n = num_samples)
    scaled_samples = qmc.scale(samples, l_bounds = domain_range[0], u_bounds = domain_range[1])
    return scaled_samples


def equivalent_classes(data_samples: np.ndarray,
                       *args) -> np.ndarray:
    """Map samples into discretized equivalent classes.

    Supported call patterns:
      equivalent_classes(samples, discretization)
      equivalent_classes(samples, forward_model, discretization)
      equivalent_classes(samples, discretization, forward_model)
    """
    if len(args) == 0:
        raise TypeError("equivalent_classes requires a discretization tuple, with an optional forward model.")

    if len(args) == 1:
        discretization = args[0]
        forward_model = None
    elif len(args) == 2:
        first, second = args
        if callable(first):
            forward_model, discretization = first, second
        elif callable(second):
            discretization, forward_model = first, second
        else:
            raise TypeError("The second argument must be either a discretization tuple or a forward model callable.")
    else:
        raise TypeError("equivalent_classes accepts at most two optional arguments beyond data_samples.")

    if forward_model is not None:
        data_samples = np.asarray(forward_model(data_samples))

    # Ensure data_samples is properly shaped
    if data_samples.ndim == 1:
        data_samples = data_samples[:, np.newaxis]

    _, datamin, cell_width, grid_shape = discretization

    index = (data_samples - datamin) / cell_width
    inbound = (index >= 0) & (index < np.array(grid_shape))
    all_inbound = inbound.all(axis=1)

    multi_index = np.floor(index).astype(int)
    flat_index = np.full(data_samples.shape[0], -1)

    if all_inbound.any():
        valid_multi_indices = multi_index[all_inbound]
        valid_flat_indices = np.ravel_multi_index(valid_multi_indices.T, grid_shape)
        flat_index[all_inbound] = valid_flat_indices

    return flat_index

