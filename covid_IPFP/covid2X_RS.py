## this is the random sampling part of the forward_SIRX model.
## We consider TWO-dimensional data which encode temporal correlation


## this is the random sampling part of the forward_SIRX model.


import sys
import os

# Get the path to the parent directory (IPFP) and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from discrete_range import discretization
from forward_SIRX import SIR_master_model
from domain import domain_samples, equivalent_classes
import json 
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta

####----- This script read and discretize the data.
#### and generate random uniform domain samples, followed by domain recovery informed by data/discreitzation
#### and last generate equivalent classes of domain samples by each forward model.



# the exact list of target FIPS codes
fips_codes = ['01081', '01125', '04005', '04013', '04019', '05143', 
    '06007', '06023', '06059', '06079', '06083', '06113', 
    '08059', '08069', '08077', '09009', '09013', '10003', 
    '12001', '12073', '13059', '16057', '17019', '17113', 
    '18105', '18141', '18157', '18167', '19013', '19103', 
    '19169', '20045', '20161', '21067', '21111', 
    '24003', '24033', '24045', '25001', '26021', '26061', 
    '26065', '26073', '26161', '27169', '28035', '28071', 
    '28105', '29001', '29019', '29161', '30001', '30031', 
    '32031', '34013', '34021', '35013', '36023', '36029', 
    '36103', '36109', '37063', 
    '37081', '37099', '37129', '37135', '37147', '37155', 
    '37183', '37189', '39017', '40027', '40119', '41029', 
    '41039', '42003', '42029', '42043', '42077', '42119', 
    '44007', '45045', '45077', '45079', '47037', '47093', 
    '47141', '48041', '48113', '48121', '48209', '48303', 
    '49021', '49049', '50007', '51059', '51121', '53037', 
    '53063', '53073', '53075', '54011', '54037', '54061', 
    '55025', '55033', '55035', '55043', '55063', '55087', 
    '55093', '55097', '56001' ]

# first, read and process the data: which is the 7 days/ 14 days /.... increment of covid cases in each county starting from the surge date

covid_datafull = pd.read_csv('college_town_data_dec22.csv',parse_dates = True, dtype = {'fips': str})
covid_start_date = pd.read_csv('omicron_surge_start_dates.csv', parse_dates = True, dtype = {'fips': str})
covid_datafull['date'] = pd.to_datetime(covid_datafull['date'])
covid_datafull['fips'] = covid_datafull['fips'].astype(str).str.zfill(5)



########## the number of data points
num_data = len(fips_codes)
dim_data = 2      # consider two dimensional data.

#########

#Create a duration of days for the increment 
duration7d = timedelta(days = 7)
duration14d = timedelta(days = 14)
duration21d = timedelta(days = 21)
duration28d = timedelta(days = 28)
duration35d = timedelta(days = 35)
duration42d = timedelta(days = 42)
duration49d = timedelta(days = 49)
duration56d = timedelta(days = 56)
duration1 = timedelta(days=  10)
duration2 = timedelta(days = 20)
duration3 = timedelta(days =30)
duration4 = timedelta(days =40)
duration5 = timedelta(days =50)
duration6 = timedelta(days =60)
duration7 = timedelta(days =70)
#duration = [duration2]
#duration = [duration1, duration2]
#duration = [duration1, duration2, duration3]
#duration = [duration1, duration2, duration3,duration4,duration5]
#duration = [duration1, duration3, duration5]

duration = [
#            [duration7d,duration14d],
#           [duration21d,duration28d],
            [duration35d,duration42d]
            ]
num_dur = len(duration)


data_discretization = []

for dur_pair in duration:
    data = np.empty((num_data,dim_data))

    # 1. Unpack the pair of timedeltas 
    dur1, dur2 = dur_pair[0], dur_pair[1]
    
    for i in range(len(fips_codes)):

        # get the surge start date for the current fips code
        surge_start_date = covid_start_date.loc[covid_start_date['fips'] == fips_codes[i], 'omicron_surge_start'].iloc[0]
        surge_start_date = pd.to_datetime(surge_start_date)
        cases_start = covid_datafull.loc[(covid_datafull['fips'] == fips_codes[i]) &(covid_datafull['date'] == surge_start_date), 'cases' ].iloc[0]
        
        # 2. Extract cases for both t1 and t1+7
        case_t1 = covid_datafull.loc[(covid_datafull['fips'] == fips_codes[i]) &(covid_datafull['date'] == surge_start_date + dur1), 'cases' ].iloc[0] 
        case_t2 = covid_datafull.loc[(covid_datafull['fips'] == fips_codes[i]) &(covid_datafull['date'] == surge_start_date + dur2), 'cases' ].iloc[0] 
        
        population = covid_datafull.loc[covid_datafull['fips'] == fips_codes[i], 'Population'].iloc[0]
        
        # 3. Assign to the respective dimensions of our 2D array
        data[i, 0] = (case_t1 - cases_start) / population
        data[i, 1] = (case_t2 - cases_start) / population
    
    

    # if data size is small, we can do jittering.
    # remove the outlier 
    # --- START OF OUTLIER REMOVAL ---
    
    # 1. Calculate the Z-scores (distance from the mean divided by standard deviation)
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    z_scores = np.abs((data - mean_vals) / std_vals)
    
    # 2. Create a mask for points that are within 2 standard deviations on BOTH axes
    # We use .all(axis=1) to ensure neither the X nor the Y coordinate is an outlier
    non_outlier_mask = (z_scores < 2).all(axis=1)
    
    # 3. Apply the mask to filter the data array
    data = data[non_outlier_mask]
    
    # (Optional) Print how many points were removed so you can monitor it
    num_removed = len(non_outlier_mask) - np.sum(non_outlier_mask)
    print(f"Removed {num_removed} outliers for duration pair: {dur1.days}d vs {dur2.days}d")
    
    
    # --- END OF OUTLIER REMOVAL ---
    

    # 4. Generate a 2D scatter plot for the current duration pair
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.7)
    plt.title(f'Normalized Cumulative Cases: {dur1.days} Days vs {dur2.days} Days')
    plt.xlabel(f'Cases at {dur1.days} days')
    plt.ylabel(f'Cases at {dur2.days} days')
    plt.grid(True)
    plt.show()
    
    
    # discretize the data
    data_discretization.append(discretization(data, num_cells = 3))

print(np.sum(data_discretization[0][0]))

with open('covidX_data_discretization.json','w') as file:
    # Convert numpy arrays to lists for JSON serialization
    data_to_save = [(disc[0].tolist()) for disc in data_discretization]
    json.dump(data_to_save, file)


# plot the histogram of the data representing the increment of cases of each county




## after obtaining the discretization of data samples,
##  we can decompose the domain samples into equivalent classes by each forward model.

### first, we create latin hypercube samples on the domain
### we assume the domain is a sufficiently large hypercube that cover all the possible parameter values
num_samples = 10000
domain_samples = domain_samples(num_samples, dim = 4, domain_range = np.array([[0,0,0.3,0],[3,3,0.998,0.002]]))

# Define the unique time points you need for all increments/ have to match the real data
eval_times = [7, 14, 21, 28, 35, 42]

#  Run the ODE solver ONCE for all samples and all time points
print("Running master ODE simulations...")

all_results = SIR_master_model(domain_samples, eval_times) 

# all_results shape is (100000, 7)
# Index mapping: 0=7d, 1=14d, 2=21d, 3=28d, 4=35d, 5=42d, 6=49d

#  Create the data samples for each duration by simply subtracting the columns/ has to match the real data 
precomputed_samples = [
#    all_results[:,[0,1]],  # 7d,14d
#    all_results[:,[2,3]],  # (21d,28d)
    all_results[:,[4,5]] 
]
    

# Loop through the precomputed intervals to get equivalent classes
indices = np.zeros((num_dur, num_samples))
for i in range(num_dur):
    # Pass the precomputed slice directly instead of a callable forward model
    index = equivalent_classes(precomputed_samples[i], data_discretization[i])    
    indices[i] = index

inbound_joint = np.where(np.all(indices > -1, axis=0))[0] 
domain_samples = domain_samples[inbound_joint] 
indices = indices[:, inbound_joint].astype(int) 

print(indices.shape)
print(domain_samples.shape)

#################### save the domain samples that match the range of data and their equivalent classes for iteration
np.save('SIRX_domain_samples.npy', domain_samples)
np.save('SIRX_indices.npy', indices)