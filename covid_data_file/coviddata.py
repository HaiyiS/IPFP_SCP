import pandas as pd

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

print("Downloading NYT data for 2021 and 2022. This may take a moment...")

# The NYT split their massive dataset by year. We need both to cover Dec-Apr.
url_2020 = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2020.csv"
url_2021 = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2021.csv"
url_2022 = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2022.csv"

# Read CSVs directly from GitHub. Force 'fips' to string to preserve leading zeros
df_2020 = pd.read_csv(url_2020, dtype={'fips': str})
df_2021 = pd.read_csv(url_2021, dtype={'fips': str})
df_2022 = pd.read_csv(url_2022, dtype={'fips': str})

# Combine the two years into one DataFrame
df_combined = pd.concat([df_2020, df_2021, df_2022])

print("Filtering data...")

# 1. Filter by Date (Dec 1, 2021 to April 30, 2022)
df_combined['date'] = pd.to_datetime(df_combined['date'])
date_mask = (df_combined['date'] >= '2020-01-21') & (df_combined['date'] <= '2022-12-30')
df_filtered = df_combined.loc[date_mask]

# 2. Filter by your target FIPS codes
df_filtered = df_filtered[df_filtered['fips'].isin(fips_codes)]

# 3. Select only the requested columns
columns_to_keep = ['date', 'county', 'state', 'fips', 'cases', 'deaths']
df_final = df_filtered[columns_to_keep]

# 4. Sort chronologically by county
df_final = df_final.sort_values(by=['fips', 'date']).reset_index(drop=True)

# Export to CSV
output_filename = "college_town_cases_deaths_Jan20_dec22.csv"
df_final.to_csv(output_filename, index=False)

print(f"Success! Extracted {len(df_final)} rows.")
print(f"Data saved to {output_filename}")