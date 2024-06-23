# Thesis Project: Developing an Early-warning system for Acute Myocardial Infractions in Catalonia

This repository contains code and data analysis for my thesis on Developing an Early-warning system for Acute Myocardial Infractions in Catalonia, Spain.

## Overview

The thesis investigates the association between environmental variables (temperature, humidity, pollution) and AMI incidence across different seasons in Catalonia. Two primary modeling approaches, SARIMAX and LSTM neural networks, are employed to analyze a dataset of hospital admissions and daily environmental data from 2010 to 2018.

## Project Structure

- **ARIMA_SARIMA_SARIMAX.ipynb**: Jupyter notebook containing code for ARIMA, SARIMA, and SARIMAX models.
- **DLNM(try)**: Directory for exploring distributed lag nonlinear models.
- **Data Cleaning**: Scripts and notebooks for data preprocessing and cleaning.
- **Exploratory_analysis.ipynb**: Notebook exploring the dataset and initial data analysis.
- **LSTM.ipynb**: Notebook containing code for LSTM model development and tuning.
- **Thesis_Zemencikova.pdf**: PDF file of the completed thesis document.
- **utils.py**: Utility functions used across different notebooks.
- **plots**: Directory containing plots generated during analysis.
- **lstm_tuning, lstm_tuning_new, sarimax_tuning**: Directories for model tuning and optimization.

## Usage

To replicate the analysis and results:

1. Clone this repository:
   ```bash
   git clone https://github.com/zemgabriela/Master_thesis.git
   cd thesis-project
