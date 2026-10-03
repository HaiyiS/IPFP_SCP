# An iterative procedure for the statistical calibration 

This repository contains the Python code and methodology for the paper "An iterative procedure for the statistical calibration". It implements IPFP method for COVID-19 data and for some simulation studies.

## The methodology
The code requires three ingredients: computer model that maps from parameter to data, multiple datasets, and initial reference parameter distribution to sample from. The procedure has three steps:
  1. For each dataset, we discretize it into probability cells. (histogram)
  2. For each computer model, evaluate it at initial samples and see if it matches a given probability cell at corresponding dataset. If it matches, assign the corresponding weight to that sample. Otherwise, eliminate the sample.
  3. Iterate the step 2 for each computer model, until the weight of samples converges.
  


## Main Repository Structure

* `covid_IPFP/`: Contains the main Python scripts for calibrating covid SIR model.
* `covid_IPFP/forward_SIRX.py`: this script code the SIR models for calibration, which contains four parameters.
* `covid_IPFP/covidX_RSS.py`: discrete data into probability cells and draw sample from initial reference parameter distribution
* `covid_IPFP/iteration_SIRX.py`: after generating samples from initial distribution, run iteration processes to reweigh samples until converged
* `covid_IPFP/forecast_SIRXnew.py`: evaluate SIR models at converged distribution (weighted samples) to forecast future time.
* `collegetowndata/`: Folder which contains downloaded datasets, including `college_town_data_dec22.csv`
* `covid19comparisoninpaper/`: folder where contains result plots.
* `simulations/`: folder contains main scripts for simulations
  

## COVID-19 data

The raw datasets used in this study can be found in: https://github.com/nytimes/covid-19-data. The filtered `college_town_data_dec22.csv` is provided in Collegetowndata.

## Environment Setup

This project relies on a lightweight stack of standard scientific computing libraries (NumPy, SciPy, Pandas) for statistical modeling and data processing, alongside standard visualization tools (Matplotlib, Seaborn). 

To ensure full reproducibility, we recommend running this code inside an isolated virtual environment. 

**Create and activate a new Conda environment and installed the dependencies**
```bash
conda create -n scp python=3.8
conda activate scp
pip install -r requirements.txt
```
## Reproducing the COVID forecasting Results

All scripts should be executed from the root directory of the project.

To execute the IPFP and generate the main results, run:
```bash
python covid_IPFP/covidX_RSS.py
python covid_IPFP/iteration_SIRX.py
```
To generate the forecast run:
``` bash
python covid_IPFP/forecast_SIRXnew.py
```

