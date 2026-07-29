import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load Data
df_cases = pd.read_csv("college_town_data_dec22.csv", 
                       parse_dates=['date'], 
                       dtype={'fips': str})

# 2. Calculate Cumulative Case Percentage
df_cases['case_pct'] = (df_cases['cases'] / df_cases['Population'])

# 3. Initialize the Plot
plt.figure(figsize=(14, 8))

# 4. Plot Individual faint lines for each county
# Get a list of unique counties to loop through
unique_counties = df_cases['fips'].unique()

for fips in unique_counties:
    # Filter the dataframe for the current county
    county_data = df_cases[df_cases['fips'] == fips]
    
    # Plot this specific county's data
    plt.plot(county_data['date'], county_data['case_pct'], 
             color='blue', alpha=0.25, linewidth=1.5)

# 5. Calculate and Plot Median Trendlines
# Group by date and calculate the median case percentage across all counties
median_data = df_cases.groupby('date')['case_pct'].mean().reset_index()

# Plot the median trendline on top (thick black line)
plt.plot(median_data['date'], median_data['case_pct'], 
         color='black', linewidth=3, label='Mean Trend')

# Create a "dummy" line just for the legend to represent the faint blue lines
plt.plot([], [], color='blue', alpha=0.5, linewidth=1.5, label='Individual Counties')

# 6. Format Aesthetics
plt.title("Cumulative COVID-19 Cases: Counties\n(university Towns: Jan 2020 - Dec 2022)", 
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Date", fontsize=12, fontweight='bold')
plt.ylabel("Cumulative Cases (proportion)", fontsize=12, fontweight='bold')

# Date formatting on X-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) 
# Note: Since the data spans 3 years, interval=3 (every 3 months) prevents X-axis crowding
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

Omi_start_date = pd.to_datetime('2021-12-01')
Omi_end_date = pd.to_datetime('2022-03-01')

plt.axvline(Omi_start_date, color='crimson', linestyle='--', linewidth=2)
plt.axvline(Omi_end_date, color='crimson', linestyle='--', linewidth=2)

# Optional: Add text to label the gap between the lines
plt.text(pd.to_datetime('2022-01-15'), plt.ylim()[1] * 0.9, 'Omicron\nSurge', 
         color='crimson', fontsize=12, fontweight='bold', ha='center')

# Grid and Legend
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12, loc='upper left', framealpha=0.9)

# Render
plt.tight_layout()
plt.savefig("cumulative_cases_covid.pdf")
plt.show()