import sys
import os

# Get the path to the parent directory (IPFP) and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from discrete_range import discretization
from forward_SIR import forward_models
from domain import domain_samples, equivalent_classes
import json 
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta

####----- This script generate simulated data by forward models and discretize the data.
#### and generate random uniform domain samples, followed by domain recovery informed by data/discreitzation
#### and last generate equivalent classes of domain samples by each forward model.


############## generate simulated data samples by pushforward normal distributed parameter samples by forward model
num_sim = 1000


# assume the trial generating distribution is normal distributed.
#lam = np.random.multivariate_normal(mean=[0, 0], cov=[[0.5, 0.7], [0.7, 1]], size=num_sim)

# assume the trial generating distribution is gaussian mixture
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.means_ = np.array([[0.4, 0.2], [0.75, 0.3]])
gmm.covariances_ = np.array([[[0.01, 0], [0, 0.002]], [[0.003, 0.001], [0.001, 0.002]]])
gmm.weights_ = np.array([0.7, 0.3])
lam = gmm.sample(n_samples= num_sim)[0]

# plot the trial generating distribution for reference
plt.figure(figsize=(8, 6))
plt.scatter(lam[:, 0], lam[:, 1], alpha=0.3, s=1, color='blue',)
plt.title('Trial Generating samples (Gaussian Mixture)')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('trial_generating_distribution.png')
plt.show()


#################### pushforward the trial generated samples by the forward model to get respective data samples 
# and its discretization
num_models = len(forward_models)
#data_samples = []
data_discretization = []
for f in forward_models:
    data = f(lam)
 #   data_samples.append(data)
    data_discretization.append(discretization(data, num_cells = 'auto'))

with open('Sim_data_discretization.json','w') as file:
    # Convert numpy arrays to lists for JSON serialization
    data_to_save = [(disc[0].tolist()) for disc in data_discretization]
    json.dump(data_to_save, file)



## after obtaining the discretization of data samples,
##  we can decompose the domain samples into equivalent classes by each forward model.

### first, we create latin hypercube samples on the domain
### we assume the domain is a sufficiently large hypercube that cover all the possible parameter values

num_samples = 50
domain_samples = domain_samples(num_samples, dim = 2 , domain_range = np.array([[0.05,0.01],[1,0.45]]))


########################## we decompose the domain samples into equivalent classes by each forward model
indices = np.zeros((num_models, num_samples))
for i in range(num_models):
    index = equivalent_classes(domain_samples, forward_models[i],data_discretization[i])    
    indices[i] = index
inbound_joint = np.where(np.all(indices > -1, axis=0))[0] 
domain_samples = domain_samples[inbound_joint] # # only keep the samples that are in the inbound of all forward models
indices = indices[:, inbound_joint].astype(int) # only keep indices that are in the inbound for all forward models.
print(indices.shape)
print(domain_samples.shape)
#################### save the domain samples that match the range of data and their equivalent classes for iteration
np.save('Sim_domain_samples.npy', domain_samples)
np.save('Sim_indices.npy', indices)
