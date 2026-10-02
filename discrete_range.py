
""" 
Discrete data points into many cells and evaluate probability of each cell
we consider histogram discretization of N-dimensional data. 
"""
import numpy as np
from typing import Tuple

def discretization(data: np.ndarray, 
                   num_cells= 'auto') -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    
    """
    Discretizes a 1D/2D/3d empirical dataset and evaluates the probability mass of each cell.
    
    Parameters:
    -----------
    data : np.ndarray
        A Nd array of continuous data points.
    num_cells : int or str, optional
        The number of cells to discretize into, or an estimator string 
        (e.g., 'auto', 'fd'). Default is 'auto'.
        
    Returns:
    --------
    prob_1d : np.ndarray
        A 1D array containing the probability mass of each cell.
    datamin : np.ndarray
        The absolute minimum boundary of the discretized domain.
    cell_width : np.nadrray
        The uniform width of each cell at each dimension.
    grid_shape: tuple
        the shape of histogram counts
    """

    if len(data) == 0:
        raise ValueError("Cannot discretize an empty data array.")
    
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    if isinstance(num_cells, str):
        bins = [np.histogram_bin_edges(data[:, i], bins=num_cells) for i in range(data.shape[1])]
    elif np.isscalar(num_cells):
        bins = int(num_cells)
    else:
        bins = [
            np.histogram_bin_edges(data[:, i], bins=bin_spec)
            if isinstance(bin_spec, str)
            else bin_spec
            for i, bin_spec in enumerate(num_cells)
        ]

    counts, bin_edges = np.histogramdd(data, bins=bins)
    prob = counts / len(data)
    
    # 1. Capture the original N-dimensional shape (e.g., (10, 10))
    grid_shape = prob.shape 
    
    # 2. Flatten the probability grid into a 1D array
    prob_1d = prob.ravel() 
    
    datamin = np.array([edges[0] for edges in bin_edges])
    cell_width = np.array([edges[1] - edges[0] for edges in bin_edges])
    
    # Return grid_shape so the domain module can calculate matching 1D indices
    return prob_1d, datamin, cell_width, grid_shape




