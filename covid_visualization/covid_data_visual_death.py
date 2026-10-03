import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 1. Load the Vaccination Snapshot Data
# Assuming this file has 'fips' and 'series_complete_pop_pct'
df_vax = pd.read_csv("college_town_vax_data_Feb.csv", dtype={'fips': str})

# If the vax file has multiple dates, take the maximum or latest rate for categorization
if 'date' in df_vax.columns:
    vax_rates = df_vax.groupby('fips')['series_complete_pop_pct'].max().reset_index()
else:
    vax_rates = df_vax[['fips', 'series_complete_pop_pct']]

# 2. Categorize the FIPS codes into High and Low cohorts
high_vax_fips = vax_rates[vax_rates['series_complete_pop_pct'] >= 68]['fips'].tolist()
low_vax_fips = vax_rates[vax_rates['series_complete_pop_pct'] <= 52]['fips'].tolist()

print(f"Found {len(high_vax_fips)} High Vax counties and {len(low_vax_fips)} Low Vax counties.")

# 3. Load the Infection Data
df_cases = pd.read_csv("college_town_data_dec22.csv", 
                       parse_dates=['date'], 
                       dtype={'fips': str})

# Calculate Cumulative death Percentage
df_cases['death_pct'] = (df_cases['deaths'] / df_cases['Population']) * 100

# 4. Split the infection data into the two cohorts
df_high = df_cases[df_cases['fips'].isin(high_vax_fips)]
df_low = df_cases[df_cases['fips'].isin(low_vax_fips)]

# 5. Initialize the Plot
plt.figure(figsize=(14, 8))

# Plot High Vax Counties (Individual faint lines)
for fips, group in df_high.groupby('fips'):
    plt.plot(group['date'], group['death_pct'], color='royalblue', alpha=0.25, linewidth=1.5)

# Plot Low Vax Counties (Individual faint lines)
for fips, group in df_low.groupby('fips'):
    plt.plot(group['date'], group['death_pct'], color='crimson', alpha=0.25, linewidth=1.5)

# 6. Calculate and Plot Median Trendlines
# High Vax Median
if not df_high.empty:
    high_median = df_high.groupby('date')['death_pct'].median().reset_index()
    plt.plot(high_median['date'], high_median['death_pct'], color='darkblue', linewidth=3.5, 
             label='High Vax (>70%) Median')

# Low Vax Median
if not df_low.empty:
    low_median = df_low.groupby('date')['death_pct'].median().reset_index()
    plt.plot(low_median['date'], low_median['death_pct'], color='darkred', linewidth=3.5, 
             label='Low Vax (<50%) Median')

# 7. Format Aesthetics
plt.title("Cumulative COVID-19 Deaths: High vs. Low Vaccination Counties\n(College Towns: Jan 2020 - Dec 2022)", 
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Date", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative Deaths (% of Population)", fontsize=12, fontweight='bold')

# Date formatting on X-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# Grid and Legend
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12, loc='upper left', framealpha=0.9)

# Render
plt.tight_layout()
plt.savefig("cumulative_deaths_high_low_vax.pdf")
plt.show()
