import pandas as pd

print("Loading your previously saved COVID data...")
# Load the dataset you just created
# Ensure fips is read as a string to keep the leading zeros (e.g., '01081')
df_covid = pd.read_csv("college_town_cases_deaths_Jan20_dec22.csv", dtype={'fips': str})

print("Fetching county population data from Johns Hopkins University...")
# JHU CSSE Lookup Table URL
jhu_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv"

# Read the JHU data
df_jhu = pd.read_csv(jhu_url)

# Filter for US counties only (removes states, territories, and international data)
df_pop = df_jhu[df_jhu['Country_Region'] == 'US'].copy()

# Drop rows where FIPS is missing
df_pop = df_pop.dropna(subset=['FIPS'])

# The JHU FIPS column is a float. We need to convert it to an integer, 
# then to a 5-digit string with leading zeros to match your COVID dataset.
df_pop['fips'] = df_pop['FIPS'].astype(int).astype(str).str.zfill(5)

# Keep only the necessary columns from the JHU dataset
df_pop = df_pop[['fips', 'Population']]

print("Merging population data with COVID data...")
# Perform a Left Join: Keeps all rows from your COVID data and matches the population based on FIPS
df_final_with_pop = pd.merge(df_covid, df_pop, on='fips', how='left')

# Check if any counties failed to match (Population is NaN)
missing_pop = df_final_with_pop[df_final_with_pop['Population'].isna()]['fips'].unique()
if len(missing_pop) > 0:
    print(f"Warning: Could not find population data for these FIPS: {missing_pop}")
else:
    print("Success! All counties matched with their population.")

# Export the final enriched dataset
output_filename = "college_town_data_with_population.csv"
df_final_with_pop.to_csv(output_filename, index=False)

print(f"File saved successfully as: {output_filename}")

# Display the first few rows to verify
print("\nPreview of the new dataset:")
print(df_final_with_pop.head())