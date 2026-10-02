""" 
after covidX_.py, we can excute the iteration procedure.
# carry out the iteration procedure using domain samples, their equivalent classes,
# data_discretization and forward models.
"""

import numpy as np
import json
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator 


# load the domain samples and their equivalent classes, and data discretization for covidX_data 

domain_samples = np.load('SIRX_domain_samples.npy') # uniform samples on the domain shape (num_samples,dim)
indices = np.load('SIRX_indices.npy') # equivalent classes shape (num_models, num_samples)
with open('covidX_data_discretization.json', 'r') as f:
    data_discretization = json.load(f) #  a list of 1d array, each array is a prob array  for each forward model

probs = [np.asarray(p, dtype=np.float64) for p in data_discretization]


# initialize the weight for each domain sample ########
num_samples = domain_samples.shape[0]


plt.figure(figsize=(8, 6))
plt.hist2d(domain_samples[:,0], domain_samples[:,1], 
           bins=70, weights=np.ones(domain_samples.shape[0])/domain_samples.shape[0], cmap='inferno',range =[[0,2], [0,2]])
cbar = plt.colorbar(label='Probability Density')
cbar.ax.set_yticklabels([])   
plt.show()

# consider intitial uniform prior 
weights = np.ones(num_samples)/ num_samples




#---------- start the iteration precedure ----------

max_iter = 9000 # maximum number of iterations for cycles
t = indices.shape[0] # number of forward models/ projections
error = 1e-4 # error threshold for convergence
eta_list = np.zeros(max_iter) # to store error at each cycle for visualization
# iteration procedure
for i in range(max_iter):
    eta = 0 # error 
    for j in range(t):
        # compute the pushforward weight by the j-th projection
        pushforward_weight = np.bincount(indices[j], weights = weights)
      #  w = np.where(pushforward_weight != 0, probs[j][:len(pushforward_weight)] / pushforward_weight, 0)
        w = np.divide(
        probs[j][:len(pushforward_weight)], 
        pushforward_weight, 
        out=np.zeros_like(pushforward_weight, dtype=float), 
        where=(pushforward_weight != 0))     # w represents the reweight term/ the fraction in the formula.
        
        eta = eta + np.sum(np.abs(w-1) * pushforward_weight)  # eta signals the distance between pushforward_weights and the probs
        weights = weights * w[indices[j]] # 
        print(np.sum(weights)) # check if the weights sum to one.

    eta_list[i] = eta
    if eta < error:
        print(f"Converged at cycle {i+1} with error: {eta}")
        break   



np.save('SIRX_final_weights.npy', weights)


# ---- Visualize the final distribution for beta and gamma ---------
plt.figure(figsize=(8, 6))

plt.hist2d(domain_samples[:,0], domain_samples[:,1], 
           bins=35, weights=weights, cmap='inferno',range =[[0,3], [0,3]])
cbar = plt.colorbar(label='Probability Density')
cbar.ax.set_yticklabels([])
plt.title(f'Heatmap solution after {i+1} cycles with error:{eta:.4f}')
plt.xlabel('$\\beta$')
plt.ylabel('$\\gamma$')

plt.savefig('heatmap_solution_SIRby3.png')
plt.show()



#-------- error convergence plot ----------
plt.figure(figsize=(8, 6))
plt.plot(eta_list[:i+1], marker = 'o', markersize=0.01)
plt.title('Error Convergence')
plt.xlabel('Cycle')
plt.ylabel('Error ( \$log_{10} \eta $ )')
plt.yscale('log')
plt.savefig('error_convergence_SIR.png')
plt.show()



