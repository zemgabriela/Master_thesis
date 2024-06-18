import pandas as pd
from datetime import date 
import holidays 
import numpy as np

# Read the CSV files
cardio = pd.read_csv('modelling_data/cardio_at_asir.csv')
environment = pd.read_csv('modelling_data/environment.csv')

# Merge the DataFrames with an indicator
combined = environment.merge(cardio, on=['date', 'at_code'], how='right', indicator=False)

# Date columns
# Get holidays for Spain for the years 2010 to 2018
es_holidays = holidays.ES(years=list(np.arange(2010, 2019)),  subdiv='CT')
holidays_df = pd.DataFrame(list(es_holidays.items()), columns=['Date', 'Holiday'])
holidays_df = holidays_df.rename(columns = {'Date':'date', 'Holiday':'is_holiday'})

holidays_df["date"] = pd.to_datetime(holidays_df["date"]) 

combined["date"] = pd.to_datetime(combined["date"])
combined = combined.merge(holidays_df, on ='date', how = 'left')
combined['is_holiday'] = np.where(combined['is_holiday'].isna(),0,1)

combined['dow'] = combined['date'].dt.day_name() #name of the day of the week
combined['day'] = combined['date'].dt.day #day of the month
combined['month'] = combined['date'].dt.month #month number
combined['year'] = combined['date'].dt.strftime('%y') #two-digit year
combined['doy'] = combined['date'].dt.dayofyear #day of the year
# Function to categorize each month into a season
def categorize_season(month):
    if month in ["05", "06", "07", "08", "09", "10"]:
        return "Summer"
    else:
        return "Winter"

# Add a new column "season" to the DataFrame based on the month
combined['season'] = combined['month'].apply(categorize_season)
# Calculate incidence rate
combined['incidence_rate'] = combined['events']/combined['population']*100000
combined['asir'] = combined['events']/combined['expected']
# Last adjustments
combined = combined.rename(columns={'Maximum Relative Humidity':'max_hum',
                          'Minimum Relative Humidity':'min_hum',
                          'Maximum Temperature':'max_temp',
                          'Minimum Temperature':'min_temp'})


combined.to_csv('modelling_data/data.csv', index = False)