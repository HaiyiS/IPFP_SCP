"""after obtaining the final weighted samples, (iteration_SIRX.py)
 we use them to forecast later stage of the pandemic using the SIR forward models
 and compare the forecasted results with real data.
this is for boxplots and forecast over time generation 
"""

import sys
import os

# Get the path to the parent directory (IPFP) and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from forward_SIRX import SIR_master_model, SIR_solution
import json 
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta
import seaborn as sns

# Set random seed for reproducible resampling
np.random.seed(42)

weights = np.load('SIRX_final_weights.npy') 
domain_samples = np.load('SIRX_domain_samples.npy')
num_samples = domain_samples.shape[0]
print(f"Sum of weights: {np.sum(weights)}")

### forecast at specific time 
times = [10,20,30,40,50,60,70,80,90]
num_time = len(times)

# PRECOMPUTE ODE FORECASTS ONCE
print("Precomputing ODE forecasts for all time points...")
all_forecasts = SIR_master_model(domain_samples, times)

# the list of target FIPS codes
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

# first, read and process the data
covid_datafull = pd.read_csv('college_town_data_dec22.csv', parse_dates=True, dtype={'fips': str})
covid_start_date = pd.read_csv('omicron_surge_start_dates.csv', parse_dates=True, dtype={'fips': str})
covid_datafull['date'] = pd.to_datetime(covid_datafull['date'])
covid_datafull['fips'] = covid_datafull['fips'].astype(str).str.zfill(5)

num_data = len(fips_codes)
data = np.empty(num_data)

# Create a duration of days for the increment 
duration = [timedelta(days=t) for t in times]
num_dur = len(duration)

mean_data = np.zeros(num_time)
mean_forecasts = np.zeros(num_time)
quantile_data = np.zeros((num_time,2))
quantile_forecasts = np.zeros((num_time,2))
quantile_range = [0.05, 0.95]

# List to gather records for the final comprehensive boxplot
boxplot_records = []

for j in range(num_time):
    for i in range(len(fips_codes)):
        # get the surge start date for the current fips code
        surge_start_date = covid_start_date.loc[covid_start_date['fips'] == fips_codes[i], 'omicron_surge_start'].iloc[0]
        surge_start_date = pd.to_datetime(surge_start_date)
        cases_start = covid_datafull.loc[(covid_datafull['fips'] == fips_codes[i]) &(covid_datafull['date'] == surge_start_date), 'cases' ].iloc[0]
        case_dur = covid_datafull.loc[(covid_datafull['fips'] == fips_codes[i]) &(covid_datafull['date'] == surge_start_date + duration[j]), 'cases' ].iloc[0] 
        population = covid_datafull.loc[covid_datafull['fips'] == fips_codes[i], 'Population'].iloc[0]
        data[i] = (case_dur - cases_start)/population
    
    # SLICE PRECOMPUTED MATRIX HERE
    forecasts = all_forecasts[:, j]
    
    min_val = np.min(data)
    max_val = np.max(data)
    bin_width = (max_val - min_val)/10

    # Individual distribution plots per time step
    plt.figure()
    sns.histplot(x=data, binwidth=bin_width, alpha=0.3, label='data', color='blue', stat="density")
    sns.kdeplot(x=forecasts, bw_method='silverman', weights=weights, fill=True, label='forecasts', color='red')
    
    mean_data[j] = np.average(data)
    mean_forecasts[j] = np.average(forecasts, weights=weights)
    quantile_data[j] = np.quantile(data, quantile_range)
    
    sorter = np.argsort(forecasts)
    sorted_samples = forecasts[sorter]
    sorted_weights = weights[sorter]
    cdf = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    quantile_forecasts[j] = np.interp(quantile_range, cdf, sorted_samples)
    
    plt.title('comparison between real data and forecasts at time {}'.format(times[j]))
    plt.xlabel('Cumulative Case Percentage Increment')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.savefig("forecast at time {}.pdf".format(times[j]))
    plt.close() # Close to preserve memory
    
    print("The mean difference at time {} is {}".format(times[j], np.abs(mean_data[j] - mean_forecasts[j])))

    # -----------------------------------------------------------------
    # DATA PREPARATION FOR THE BOXPLOT
    # -----------------------------------------------------------------
    # Normalize weights for random choice selection
    norm_weights = weights / np.sum(weights)
    # Resample forecasts to convert the weighted distribution into unweighted observations
    forecasts_resampled = np.random.choice(forecasts, size=30000, p=norm_weights)
    # Store real data observations
    for val in data:
        boxplot_records.append({'Time': times[j], 'Value': val, 'Type': 'Real Data'})
    # Store resampled forecast observations
    for val in forecasts_resampled:
        boxplot_records.append({'Time': times[j], 'Value': val, 'Type': 'Forecast'})

# Convert all records to a tidy DataFrame
df_boxplot = pd.DataFrame(boxplot_records)

# save the datafram/ the index=False argument prevents pandas from writing row numbers to the file
df_boxplot.to_csv('boxplot_data56.csv', index=False)
# -----------------------------------------------------------------
# PLOT 1: THE NEW BOXPLOT COMPARISON
# -----------------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='Time', 
    y='Value', 
    hue='Type', 
    data=df_boxplot, 
    palette={'Real Data': '#5D9CEC', 'Forecast': "#FC5151"},
    width=0.6,
    showfliers=False  # Size of outlier points
)

plt.title('Forecast vs Real Data at Specific Time Points')
plt.xlabel('Time (Days since surge)')
plt.ylabel('Change of cumulative cases (proportion)')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("forecast_vs_data_boxplot.pdf")
plt.show()

# -----------------------------------------------------------------
# PLOT 2: ORIGINAL MEAN AND UNCERTAINTY PLOT
# -----------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(times, mean_data, color='blue', linewidth=2, label='Data Mean')
plt.plot(times, mean_forecasts, color='red', linewidth=2, linestyle='--', label='Forecast Mean')

plt.fill_between(times, quantile_data[:,0], quantile_data[:,1], color='blue', alpha=0.15, label='Data 90% CI')
plt.fill_between(times, quantile_forecasts[:,0], quantile_forecasts[:,1], color='red', alpha=0.15, label='Forecast 90% PI')

plt.xlabel('Time')
plt.ylabel('Change of cumulative cases (proportion)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left') 
plt.tight_layout()
plt.savefig("mean and uncertainty_forecastings.pdf")
plt.show()