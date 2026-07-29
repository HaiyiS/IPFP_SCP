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

####----- This script read and discretize the data.
#### and generate random uniform domain samples, followed by domain recovery informed by data/discreitzation
#### and last generate equivalent classes of domain samples by each forward model.



# Your exact list of target FIPS codes
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

# first, read and process the data: which is the 7 days/ 14 days increment of covid cases in each county starting from the surge date

covid_datafull = pd.read_csv('college_town_data_dec22.csv',parse_dates = True, dtype = {'fips': str})
covid_start_date = pd.read_csv('omicron_surge_start_dates.csv', parse_dates = True, dtype = {'fips': str})
covid_datafull['date'] = pd.to_datetime(covid_datafull['date'])
covid_datafull['fips'] = covid_datafull['fips'].astype(str).str.zfill(5)
# the number of data points
num_data = len(fips_codes)
data = np.empty(num_data)

#Create a duration of days for the increment 

duration1 = timedelta(days=  20)
duration2 = timedelta(days = 35)
duration3 = timedelta(days =50)

#duration = [duration1, duration2]
duration = [duration1, duration2, duration3]
#duration = [duration2]
num_dur = len(duration)


data_discretization = []

for dur in duration:
    for i in range(len(fips_codes)):

        # get the surge start date for the current fips code
        surge_start_date = covid_start_date.loc[covid_start_date['fips'] == fips_codes[i], 'omicron_surge_start'].iloc[0]
        surge_start_date = pd.to_datetime(surge_start_date)
        cases_start = covid_datafull.loc[(covid_datafull['fips'] == fips_codes[i]) &(covid_datafull['date'] == surge_start_date), 'cases' ].iloc[0]
        case_dur = covid_datafull.loc[(covid_datafull['fips'] == fips_codes[i]) &(covid_datafull['date'] == surge_start_date + dur), 'cases' ].iloc[0] 
        population = covid_datafull.loc[covid_datafull['fips'] == fips_codes[i], 'Population'].iloc[0]
        data[i] = (case_dur - cases_start)/population
    
    hist = plt.hist(data, bins=10, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Cumulative Case Percentage Increments')
    plt.xlabel('Cumulative Case Percentage Increment')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    # discretize the data
    data_discretization.append(discretization(data, num_cells = 6))

print(np.sum(data_discretization[1][0]))

with open('covid_data_discretization.json','w') as file:
    # Convert numpy arrays to lists for JSON serialization
    data_to_save = [(disc[0].tolist()) for disc in data_discretization]
    json.dump(data_to_save, file)


# plot the histogram of the data representing the increment of cases of each county




## after obtaining the discretization of data samples,
##  we can decompose the domain samples into equivalent classes by each forward model.

### first, we create latin hypercube samples on the domain
### we assume the domain is a sufficiently large hypercube that cover all the possible parameter values
num_samples = 50000
domain_samples = domain_samples(num_samples, dim = 2 , domain_range = np.array([[0,0],[1,1]]))


########################## we decompose the domain samples into equivalent classes by each forward model
indices = np.zeros((num_dur, num_samples))
for i in range(num_dur):
    index = equivalent_classes(domain_samples, forward_models[i],data_discretization[i])    
    indices[i] = index
inbound_joint = np.where(np.all(indices > -1, axis=0))[0] 
domain_samples = domain_samples[inbound_joint] # # only keep the samples that are in the inbound of all forward models
indices = indices[:, inbound_joint].astype(int) # only keep indices that are in the inbound for all forward models.
print(indices.shape)
print(domain_samples.shape)
#################### save the domain samples that match the range of data and their equivalent classes for iteration
np.save('SIR_domain_samples.npy', domain_samples)
np.save('SIR_indices.npy', indices)
