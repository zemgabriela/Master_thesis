# =============== LIBRARIES  ===============
import pandas as pd
# =============== LOAD  ===============
daily_data = pd.read_csv('../modelling_data/data.csv')
# =============== SELECT COLUMNS  ===============
columns = ['date', 'at_code', 'max_hum', 'min_hum', 'max_temp', 'min_temp',
            'max_C6H6', 'max_CO', 'max_Cl2',
            'max_H2S', 'max_Hg', 'max_NO', 'max_NO2', 'max_NOX', 'max_O3',
            'max_PM1', 'max_PM10', 'max_PM2.5', 'max_SO2',
            'mean_C6H6', 'mean_CO', 'mean_Cl2', 'mean_H2S', 'mean_Hg', 'mean_NO',
            'mean_NO2', 'mean_NOX', 'mean_O3', 'mean_PM1', 'mean_PM10',
            'mean_PM2.5', 'mean_SO2', 'min_C6H6', 'min_CO', 'min_Cl2', 'min_H2S',
            'min_Hg', 'min_NO', 'min_NO2', 'min_NOX', 'min_O3', 'min_PM1',
            'min_PM10', 'min_PM2.5', 'min_SO2', 'events', 'is_holiday', 'incidence_rate']
daily_data = daily_data[columns]
# =============== CONVERT DATATYPE  ===============
daily_data['date'] = pd.to_datetime(daily_data['date'])

# =============== DEFINE FUNCTIONS  ===============
# Define custom aggregation function
def custom_agg(df):
    agg_dict = {col: 'mean' for col in df.columns if col not in ['events', 'is_holiday']}
    agg_dict.update({'events': 'sum', 'is_holiday': 'sum'})
    return df.agg(agg_dict, skipna=True)

# Resample and aggregate data
def resample_and_aggregate(df, freq):
    resampled_df = df.resample(freq).apply(custom_agg)
    resampled_df['month'] = resampled_df.index.get_level_values('date').month
    resampled_df['year'] = resampled_df.index.get_level_values('date').year
    resampled_df['events'] = resampled_df['events'].astype(int)
    resampled_df['is_holiday'] = resampled_df['is_holiday'].astype(int)
    return resampled_df

# Set the date as the index for resampling
daily_data.set_index('date', inplace=True)

# =============== RESAMPLE  ===============
# Resample weekly and monthly
weekly_resampled_data = daily_data.groupby('at_code').apply(resample_and_aggregate, 'W').reset_index()
monthly_resampled_data = daily_data.groupby('at_code').apply(resample_and_aggregate, 'M').reset_index()
weekly_resampled_data.to_csv('../modelling_data/weekly_data.csv',  index=False)
monthly_resampled_data.to_csv('../modelling_data/monthly_data.csv', index=False)