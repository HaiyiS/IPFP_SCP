
# after random_sampling.py, we can excute the iteration procedure.
# carry out the iteration procedure using domain samples, their equivalent classes,
#  data_discretization and forward models.

import numpy as np
import json
from scipy.stats import multivariate_normal
from forward_model import forward_models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import LogNorm

domain_samples = np.load('domain_samples.npy') # uniform samples on the domain shape (num_samples,dim)
indices = np.load('indices.npy') # equivalent classes shape (num_models, num_samples)
with open('data_discretization.json', 'r') as f:
    data_discretization = json.load(f) # a list of tuples, each tuple contains (prob, datamin, cell_width) for each forward model

probs = [np.asarray(p, dtype=np.float64) for p in data_discretization]

print(np.shape(domain_samples))
print(np.shape(indices))
print(len(probs[0]), len(probs[1]), len(probs[2]), len(probs[3]), len(probs[4]), len(probs[5]), len(probs[6]), len(probs[7]), len(probs[8]), len(probs[9]), len(probs[10]))
print(np.max(indices[0]), np.max(indices[1]), np.max(indices[2]), np.max(indices[3]), np.max(indices[4]), np.max(indices[5]), np.max(indices[6]), np.max(indices[7]), np.max(indices[8]), np.max(indices[9]), np.max(indices[10]    ))


######### initialize the weight for each domain sample ########
num_samples = domain_samples.shape[0]
########here we consider uniform weight for initial prior, which maximize entropy/ can consider other weights
#weights = np.ones(num_samples)/ num_samples 


###### Consider initial Gaussian mixture weight with true mean value but wrong covariance value 
#mean1 = [-1,-1]
#mean2 = [1,1]
#cov1 = [[2,0],[0,2]]
#cov2 = [[2,0],[0,2]]
#weights = multivariate_normal.pdf(domain_samples,mean = mean1,cov = cov1)
#+ multivariate_normal.pdf(domain_samples,mean = mean2,cov = cov2)
#weights = weights / np.sum(weights) # normalize the weights to sum to one.

# consider intitial gaussian mixture weight with wrong mean and wrong covariance value
# roughly two standard deviation away from the true mean value....
mean1 = [0,0]
#mean2 = [4,4]
cov1 = [[1,0],[0,1]]
#cov2 = [[1,0],[0,1]]
weights = multivariate_normal.pdf(domain_samples,mean = mean1,cov = cov1)
# + multivariate_normal.pdf(domain_samples,mean = mean2,cov = cov2)
weights = weights / np.sum(weights) # normalize the weights to sum to one.



########## start the iteration precedure ##########
max_iter = 2000 # maximum number of iterations for cycles
t = len(forward_models) # number of forward models/ projections
error = 1e-2 # error threshold for convergence
eta_list = np.zeros(max_iter) # to store error at each cycle for visualization
# iteration procedure
for i in range(max_iter):
    eta = 0 # error 
    for j in range(t):
        # compute the pushforward weight by the j-th projection
        pushforward_weight = np.bincount(indices[j], weights = weights)
        w = np.where(pushforward_weight != 0, probs[j][:len(pushforward_weight)] / pushforward_weight, 0)
        eta = eta + np.sum(np.abs(w-1) * pushforward_weight) 
        weights = weights * w[indices[j]] # update weights
        print(np.sum(weights)) # check if the weights sum to one.
    eta_list[i] = eta
    if eta < error:
        print(f"Converged at cycle {i+1} with error: {eta}")
        break   

###### Visualize the final distribution ########
plt.figure(figsize=(8, 6))

plt.hist2d(domain_samples[:,0], domain_samples[:,1], 
           bins=70, weights=weights, cmap='inferno',range =[[-4,4], [-6,6]])
cbar = plt.colorbar(label='Probability Density')
cbar.ax.set_yticklabels([])
plt.title(f'Heatmap solution after {i+1} cycles with error:{eta:.4f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('heatmap_solution_exp_g1-7_wrongprior_cycle8.png')
plt.show()

###### visualize the final distribution by scatter plot ########
#plt.figure(figsize=(8, 6))
#plt.scatter(domain_samples[:,0], domain_samples[:,1], c=weights, cmap='plasma', s=30, edgecolors='none', alpha=1,
#            )
#cbar = plt.colorbar(label='Probability Density')
#cbar.ax.set_yticklabels([])
#plt.title(f'Scatter solution of the IPFP_f1-18 (nonLinear) with {i+1} cycles, error:{eta:.4f}')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.savefig('scatter_solution_f1-18 (nonLinear).png')
#plt.show()

############################# 

######Sort the indices based on weights
#sort_idx = np.argsort(weights)
#sorted_samples = domain_samples[sort_idx]
#sorted_weights = weights[sort_idx]

#plt.figure(figsize=(8, 6))
####### Plot the sorted data
#plt.scatter(sorted_samples[:,0], sorted_samples[:,1], c=sorted_weights, cmap='magma', 
#            s=30, edgecolors='black', alpha=1)


############################### visualize the solution by KDE #############
#plt.figure(figsize=(8, 6))
#sns.kdeplot(x=domain_samples[:, 0], y=domain_samples[:, 1], weights=weights, color = 
#    'blue', fill=True, cmap='viridis' )
#plt.title(f'density solution of the IPFP_g with {i+1} cycles, error:{eta:.4f}')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.savefig('kde_solution_g (nonLinear).png')
#plt.show()

######### error convergence plot ##########
plt.figure(figsize=(8, 6))
plt.plot(eta_list[:i+1], marker = 'o', markersize=0.01)
plt.title('Error Convergence')
plt.xlabel('Cycle')
plt.ylabel('Error ( \$log_{10} \eta $ )')
plt.yscale('log')
plt.savefig('error_convergence_exp_g1_goodprior.png')
plt.show()

