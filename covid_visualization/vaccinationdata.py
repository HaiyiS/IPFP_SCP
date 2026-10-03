import pandas as pd
import requests

# List of FIPS codes for the requested college town counties
# Note: Stored as strings to preserve leading zeros
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

# Format FIPS codes for the API SQL-like query
fips_query_string = "','".join(fips_codes)

# CDC SODA API Endpoint for County Vaccination Data
base_url = "https://data.cdc.gov/resource/8xkx-amqh.json"

# Construct the query parameters
# Date range: Dec 1, 2021 to April 30, 2022
query = (
    f"$where=date between '2022-2-01T00:00:00' and '2022-02-01T23:59:59' "
    f"AND fips in ('{fips_query_string}')"
    f"&$limit=10000" # Ensure the limit is high enough to capture all daily rows
)

url = f"{base_url}?{query}"

print("Fetching data from CDC API...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    
    # Keep only the essential columns for modeling
    columns_to_keep = [
        'date', 'fips', 'recip_county', 'recip_state', 
        'series_complete_pop_pct', # Fully vaccinated percentage
        'administered_dose1_pop_pct', # At least one dose percentage
        'booster_doses_vax_pct' # Booster percentage (crucial for Omicron wave modeling)
    ]
    
    # Filter columns if they exist in the returned payload
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]
    
    # Convert percentages from string to numeric
    numeric_cols = ['series_complete_pop_pct', 'administered_dose1_pop_pct', 'booster_doses_vax_pct']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Sort by the highest vaccination percentage for each county 
    df = df.sort_values(by=['administered_dose1_pop_pct'], ascending=[False])
    
    print(f"Successfully retrieved {len(df)} records.")



    
    # Save to a local CSV file for immediate modeling use
    df.to_csv('college_town_vax_data_Feb2.csv', index=False)
    print("Data saved to 'college_town_vax_data_Feb2.csv'")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")