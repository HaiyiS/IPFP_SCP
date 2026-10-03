import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("Loading data...")

# 1. Load the Rolling Averages Data (Jan 2020 - Apr 2022)
df_cases_deaths = pd.read_csv("college_town_official_rolling_averages_jan20_apr22.csv",
                              parse_dates=['date'], dtype={'fips': str})

# 2. Load the Vaccination Data 
df_vax = pd.read_csv("college_town_vax_data_Feb.csv", dtype={'fips': str})

if 'date' in df_vax.columns:
    vax_rates = df_vax.groupby('fips')['series_complete_pop_pct'].max().reset_index()
else:
    vax_rates = df_vax[['fips', 'series_complete_pop_pct']]

# 3. Categorize High (>70%) vs Low (<50%) Vaccination Counties
high_vax_fips = vax_rates[vax_rates['series_complete_pop_pct'] >= 70]['fips'].tolist()
low_vax_fips = vax_rates[vax_rates['series_complete_pop_pct'] <= 50]['fips'].tolist()

print(f"Comparing {len(high_vax_fips)} High-Vax counties vs {len(low_vax_fips)} Low-Vax counties.")

# 4. Fetch Population Data
print("Fetching population lookup table...")
jhu_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv"
df_pop = pd.read_csv(jhu_url)
df_pop = df_pop[df_pop['Country_Region'] == 'US'].dropna(subset=['FIPS']).copy()
df_pop['fips'] = df_pop['FIPS'].astype(int).astype(str).str.zfill(5)
df_pop = df_pop[['fips', 'Population']]

# Merge population into the main dataset
df_main = pd.merge(df_cases_deaths, df_pop, on='fips', how='left')
df_main = df_main.dropna(subset=['Population']) 

print("Calculating daily percentages and medians...")

# 5. Calculate Daily Rates as a PERCENTAGE of the population
df_main['cases_pct'] = (df_main['cases_avg'] / df_main['Population']) * 100
df_main['deaths_pct'] = (df_main['deaths_avg'] / df_main['Population']) * 100

# 6. Split into Cohorts
df_high = df_main[df_main['fips'].isin(high_vax_fips)]
df_low = df_main[df_main['fips'].isin(low_vax_fips)]

# 7. Calculate Medians over time
high_median = df_high.groupby('date')[['cases_pct', 'deaths_pct']].mean().reset_index()
low_median = df_low.groupby('date')[['cases_pct', 'deaths_pct']].mean().reset_index()

print("Generating plots...")

# ==========================================
# FIGURE 1: DAILY CASES PERCENTAGE
# ==========================================
plt.figure(figsize=(12, 6))

# Plot Individual High Vax Counties (Faint Blue)
for fips, group in df_high.groupby('fips'):
    plt.plot(group['date'], group['cases_pct'], color='cornflowerblue', alpha=0.15, linewidth=1)

# Plot Individual Low Vax Counties (Faint Red)
for fips, group in df_low.groupby('fips'):
    plt.plot(group['date'], group['cases_pct'], color='lightcoral', alpha=0.15, linewidth=1)

# Plot the Medians ON TOP (Bold)
plt.plot(high_median['date'], high_median['cases_pct'], color='darkblue', linewidth=3, label='High Vax (>70%) Median')
plt.plot(low_median['date'], low_median['cases_pct'], color='darkred', linewidth=3, label='Low Vax (<50%) Median')

plt.title("Daily New COVID-19 Cases (% of Population)", fontsize=15, fontweight='bold')
plt.ylabel("Daily Cases (%)", fontsize=12, fontweight='bold')
plt.xlabel("Date", fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper left')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_cases_percentage.png", dpi=300)
# ==========================================
# FIGURE 2: DAILY DEATHS PERCENTAGE
# ==========================================
plt.figure(figsize=(12, 6))

# Plot Individual High Vax Counties (Faint Blue)
for fips, group in df_high.groupby('fips'):
    plt.plot(group['date'], group['deaths_pct'], color='cornflowerblue', alpha=0.15, linewidth=1)

# Plot Individual Low Vax Counties (Faint Red)
for fips, group in df_low.groupby('fips'):
    plt.plot(group['date'], group['deaths_pct'], color='lightcoral', alpha=0.15, linewidth=1)

# Plot the Medians ON TOP (Bold)
plt.plot(high_median['date'], high_median['deaths_pct'], color='darkblue', linewidth=3, label='High Vax (>70%) Median')
plt.plot(low_median['date'], low_median['deaths_pct'], color='darkred', linewidth=3, label='Low Vax (<50%) Median')

plt.title("Daily COVID-19 Deaths (% of Population)", fontsize=15, fontweight='bold')
plt.ylabel("Daily Deaths (%)", fontsize=12, fontweight='bold')
plt.xlabel("Date", fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper left')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_deaths_percentage.png", dpi=300)
# Render both separate figures
plt.show()