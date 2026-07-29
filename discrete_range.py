# Discrete data points into many cells and evaluate probability of each cell

# For now I only consider discretization of one-dimensional data. 
import numpy as np
from typing import  Tuple

def discretization(data: np.ndarray, 
                      num_cells = 'auto') -> Tuple[np.ndarray, float, float]:
    """
    Discretizes a 1D empirical dataset and evaluates the probability mass of each cell.
    
    Parameters:
    -----------
    data : np.ndarray
        A 1D array of continuous data points.
    num_cells : int or str, optional
        The number of cells to discretize into, or an estimator string 
        (e.g., 'auto', 'fd'). Default is 'auto'.
        
    Returns:
    --------
    prob : np.ndarray
        A 1D array containing the probability mass of each cell.
    datamin : float
        The absolute minimum boundary of the discretized domain.
    cell_width : float
        The uniform width of each cell.
    """
    # Safety check to prevent division by zero
    if len(data) == 0:
        raise ValueError("Cannot discretize an empty data array.")
        
    # The core logic
    counts, bin_edges = np.histogram(data, bins=num_cells)
    prob = counts / len(data)
    
    # Extracting spatial parameters
    datamin = bin_edges[0]
    cell_width = bin_edges[1] - bin_edges[0]
    
    return prob, datamin, cell_width







