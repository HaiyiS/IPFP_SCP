import pandas as pd
import numpy as np
# compute the start of the omicron surge for a given county

# '01081', '01125', '04005', '04013', '04019', '05143', 
#'06007', '06023', '06059', '06079', '06083', '06113', 

df = pd.read_csv("college_town_official_rolling_averages_jan20_apr22.csv", 
                 parse_dates=['date'], 
                 dtype={'fips': str})

def find_surge_peak(county_df, start_window, end_window):
    # 1. Isolate the time period between the two expected surges
    mask = (county_df['date'] >= start_window) & (county_df['date'] <= end_window)
    valley_period = county_df.loc[mask]
    
    # 2. Find the row with the maximum 7-day average cases
    nadir_row = valley_period.loc[valley_period['cases_avg'].idxmax()]
    
    peak_date = nadir_row['date']
    baseline_cases = nadir_row['cases_avg']
    
    print(f"Omicron Surge Peak: {peak_date}")
    print(f"Starting baseline (I_peak roughly based on this): {baseline_cases} cases")
    
    return peak_date



def find_surge_start_sustained_growth(county_df, search_start_date, consecutive_days=7):
    """
    Finds the t=0 start date of a surge based on N consecutive days of strictly increasing cases.
    """
    # Isolate the data after your starting search window 
    # (e.g., search after Nov 1, 2021 to find the Omicron surge)
    search_df = county_df[county_df['date'] >= search_start_date].copy()
    search_df = search_df.sort_values('date').reset_index(drop=True)
    
    # Step A: Check if each day's average is strictly greater than the previous day's
    search_df['is_increasing'] = search_df['cases_avg'].diff() >= 0
    
    # Step B: Check if the NEXT 7 days are all 'True'
    # We do this by looking forward (shifting backwards negatively) and summing the Trues
    future_growth_sum = sum(search_df['is_increasing'].shift(-i) for i in range(1, consecutive_days + 1))
    
    # Step C: The surge starts on the first date where the next N days sum perfectly to N
    valid_starts = search_df[future_growth_sum == consecutive_days]
    
    if not valid_starts.empty:
        # Return the very first date that meets the criteria
        start_date = valid_starts.iloc[0]['date']
        baseline_cases = valid_starts.iloc[0]['cases_avg']
        
        print(f"Surge Start Date (t=0) found: {start_date.strftime('%Y-%m-%d')}")
        print(f"Starting Cases (I_0 baseline): {baseline_cases}")
        return start_date
    else:
        print(f"No sustained {consecutive_days}-day growth found after {search_start_date}.")
        return None

# ==========================================
# Test the method on a single county
# ==========================================

# Example: Filter for Tompkins County, NY (FIPS 36109)
my_county = df[df['fips'] == '06113']


# '01081', '01125', '04005', '04013', '04019', '05143', 
#'06007', '06023', '06059', '06079', '06083', '06113', 

print("Executing User-Proposed Method (7-Day Sustained Growth)...")
# Start searching for the Omicron wave after the Delta wave settled (e.g., Nov 15, 2021)

# If you want to relax the strictness of your method, simply change the parameter:
# t_zero_relaxed = find_surge_start_sustained_growth(my_county, '2021-11-15', consecutive_days=5)

# Look for the peak point between Jan 1 and Apr 15
t_peak = find_surge_peak(my_county, '2021-10-01', '2022-04-01')


fips_codes = [
    '01081', '01125', '04005', '04013', '04019', '05143', 
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
    '55093', '55097', '56001'
]

osurge = []

for fips in fips_codes:
    county_df = df[df['fips'] == fips]
    osurge.append(find_surge_start_sustained_growth(county_df, search_start_date='2021-12-01', consecutive_days=1).date())


data = {'fips': fips_codes, 'omicron_surge_start': osurge}

df_surge_starts = pd.DataFrame(data)
df_surge_starts.to_csv("omicron_surge_start_dates.csv", index=False)

    
