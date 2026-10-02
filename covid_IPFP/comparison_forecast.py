import sys
import os

# Get the path to the parent directory (IPFP) and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_boxplot35 = pd.read_csv('boxplot_data35.csv')
df_boxplot42 = pd.read_csv('boxplot_data42.csv')
df_boxplot49 = pd.read_csv('boxplot_data49.csv')
df_boxplot56 = pd.read_csv('boxplot_data56.csv')
# -----------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------
# 1. Extract 'Real Data' just once (from the first dataframe)
df_real = df_boxplot35[df_boxplot35['Type'] == 'Real Data'].copy()

# 2. Extract the forecast from the first dataframe 
df_forecast35 = df_boxplot35[df_boxplot35['Type'] != 'Real Data'].copy()
df_forecast35['Type'] = 'Forecast35d'  
# 3. 
df_forecast42 = df_boxplot42[df_boxplot42['Type'] != 'Real Data'].copy()
df_forecast42['Type'] = 'Forecast42d'  # Feel free to rename to 'Forecast 2'
# 4.
df_forecast49 = df_boxplot49[df_boxplot49['Type'] != 'Real Data'].copy()
df_forecast49['Type'] = 'Forecast49d'  
#5.
df_forecast56 = df_boxplot56[df_boxplot56['Type'] != 'Real Data'].copy()
df_forecast56['Type'] = 'Forecast56d'  
# 6. Combine all three pieces into a single dataframe
df_boxplot = pd.concat([df_real, df_forecast35, df_forecast42, df_forecast49, df_forecast56], ignore_index=True)


# -----------------------------------------------------------------
# PLOT：THE NEW BOXPLOT COMPARISON
# -----------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Notice the palette now includes three colors for our three distinct 'Types'
sns.boxplot(
    x='Time', 
    y='Value', 
    hue='Type', 
    data=df_boxplot, 
    palette={
        'Real Data': '#5D9CEC',   # Blue
        'Forecast35d': "#E5F552", 
        'Forecast42d': "#ED9507",
        'Forecast49d': "#AB5E27", 
        'Forecast56d': "#FC0303"
    },
    width=0.6,
    showfliers=False
)

plt.title('Forecasts vs Real Data at Specific Time Points')
plt.xlabel('Time (Days since surge)')
plt.ylabel('Change of cumulative cases (proportion)')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("forecast_vs_data_boxplot.pdf")
plt.show()