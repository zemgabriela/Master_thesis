import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def data_split(data, method):
    if method == 'min':
        data = data[['at_code', 'events', 'asir', 'min_temp', 'min_hum', 'is_holiday', 'month', 'year', 
                     'min_PM10', 'min_O3', 'min_NO2', 'min_SO2', 'min_CO']].rename(
            columns={'min_temp': 'temp', 'min_hum': 'hum', 'min_PM10': 'pm10', 'min_O3': 'o3',
                     'min_NO2': 'no2', 'min_SO2': 'so2', 'min_CO': 'co'})
    elif method == 'max':
        data = data[['at_code', 'events', 'asir', 'max_temp', 'max_hum', 'is_holiday', 'month', 'year', 
                     'max_PM10', 'max_O3', 'max_NO2', 'max_SO2', 'max_CO']].rename(
            columns={'max_temp': 'temp', 'max_hum': 'hum', 'max_PM10': 'pm10', 'max_O3': 'o3',
                     'max_NO2': 'no2', 'max_SO2': 'so2', 'max_CO': 'co'})
    elif method == 'mean':
        data['mean_temp'] = (data['max_temp'] + data['min_temp']) / 2
        data['mean_hum'] = (data['max_hum'] + data['min_hum']) / 2
        data = data[['at_code', 'events', 'asir', 'mean_temp', 'mean_hum', 'is_holiday', 'month', 'year', 
                     'mean_PM10', 'mean_O3', 'mean_NO2', 'mean_SO2', 'mean_CO']].rename(
            columns={'mean_temp': 'temp', 'mean_hum': 'hum', 'mean_PM10': 'pm10', 'mean_O3': 'o3',
                     'mean_NO2': 'no2', 'mean_SO2': 'so2', 'mean_CO': 'co'})
    elif method == 'mix':
        data = data[['at_code', 'events', 'asir', 'max_temp', 'min_hum', 'is_holiday', 'month', 'year', 
                     'mean_PM10', 'mean_O3', 'mean_NO2', 'mean_SO2', 'mean_CO']].rename(
            columns={'max_temp': 'temp', 'min_hum': 'hum', 'mean_PM10': 'pm10', 'mean_O3': 'o3',
                     'mean_NO2': 'no2', 'mean_SO2': 'so2', 'mean_CO': 'co'})
    else:
        raise ValueError("Invalid method. Choose from 'min', 'max', 'mean', 'mix'.")
    
    return data

# Data Split and Plot Function
def data_missing_plot(file_path, at_codes):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(len(at_codes), 1, figsize=(14, 8 * len(at_codes)), sharex=True)
    
    # Ensure axes is always iterable
    if len(at_codes) == 1:
        axes = [axes]

    for i, at_code in enumerate(at_codes):
        # Filter the data for the current AT code
        df_f_0_f = df[df['at_code'] == at_code][['max_C6H6','max_CO','max_Cl2','max_H2S','max_Hg','max_NO','max_NO2','max_NOX','max_O3','max_PM1','max_PM10','max_PM2.5','max_SO2']].copy()
        df_f_0_f.columns = ['C6H6','CO','Cl2','H2S','Hg','NO','NO2','NOX','O3','PM1','PM10','PM2.5','SO2']
        
        # Create DataFrame for missing values
        df_missing = pd.DataFrame(df_f_0_f.isnull().sum(), columns=["# Missing Data"])
        df_missing["% Missing"] = round(df_missing["# Missing Data"] / len(df_f_0_f) * 100, 2)

        # Filter out variables with 0% missing data if necessary
        # df_missing = df_missing[df_missing["% Missing"] > 0]

        # Get variables and values for the specified columns
        variables = df_missing.index
        values = df_missing.loc[variables, "# Missing Data"]
        percentages = df_missing.loc[variables, "% Missing"]

        # Nicer colors
        present_color = 'skyblue'
        missing_color = 'salmon'

        # Plot on the corresponding subplot
        ax = axes[i]
        ax.bar(variables, values, color=missing_color, label='Missing Data')
        ax.bar(variables, len(df_f_0_f) - values, bottom=values, color=present_color, label='Present Data')

        # Labels for percentage of missing data
        for j, (percentage, value) in enumerate(zip(percentages, values)):
            if percentage > 0:  # Only add the label if the percentage is greater than 0%
                ax.text(j, value / 2, f"{percentage:.0f}%", ha='center', va='center', color='black', fontsize=8, rotation=90)

        # Plot configuration
        ax.set_ylabel('Number of Data Points', fontsize=12)
        ax.set_title(f'Number of Present and Missing Data Points by Variable for {at_code}', fontsize=14)
        ax.legend(fontsize=12)
    
    # Show plot
    plt.xlabel('Pollutant', fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'plots/missing_contaminants.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    return 

def time_series_decompose_and_plot(df, at_code, time_series_column, period=52):
    # Decompose the time series
    decomposition = seasonal_decompose(df[time_series_column], model='additive', period=period)  

    # Plot the decomposition
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(decomposition.trend, label='Trend')
    plt.title('Trend Component')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 2)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.title('Seasonal Component')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 3)
    plt.plot(decomposition.resid, label='Residual')
    plt.title('Residual Component')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 4)
    plt.plot(df[time_series_column], label='Original')
    plt.title('Original Time Series')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'plots/{at_code}_asir_decomposition.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    return

def corrMat(df, at_code):
    
    # Calculate the correlation matrix
    corr_mat = df.corr().round(2)
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    np.fill_diagonal(mask, False)
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 6))
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_mat, mask=mask, vmin=-1, vmax=1, center=0, 
                cmap='crest', square=True, linewidths=2, annot=True, cbar=False,
                xticklabels = True, yticklabels=True)
    
    # Display the heatmap
    plt.savefig(f'plots/{at_code}_correlation_plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    return

def train_test_split(train, validation, at_code):
    # Plot
    fig, ax = plt.subplots(figsize=(7, 3))
    train['asir'].plot(ax=ax, label='train')  # Plot the 'asir' column of the training set
    validation['asir'].plot(ax=ax, label='test')  # Plot the 'asir' column of the validation set
    ax.set_title(f'Weekly scaled ASIR train/test split - {at_code}')
    ax.legend()
    plt.grid(True)
    plt.ylabel('ASIR scaled')
    plt.savefig(f'plots/{at_code}_train_test_scaled_split.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    return

def plot_acf_pacf(df_f_scaled, at_code):
    # Plot ACF and PACF of target
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plot_acf(df_f_scaled['asir'], ax=plt.gca(), lags=50)
    plt.title('Autocorrelation Function')

    plt.subplot(122)
    plot_pacf(df_f_scaled['asir'], ax=plt.gca(), lags=50)
    plt.title('Partial Autocorrelation Function')

    plt.tight_layout()
    plt.savefig(f'plots/{at_code}_acf_pacf.pdf', format='pdf', bbox_inches='tight')
    plt.show()