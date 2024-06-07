import pandas as pd

# Read the CSV files
cardio = pd.read_csv('modelling_data/cardio_at_asir.csv')
environment = pd.read_csv('modelling_data/environment.csv')

# Merge the DataFrames with an indicator
combined = environment.merge(cardio, on=['date', 'at_code'], how='right', indicator=False)
combined.to_csv('modelling_data/data.csv', index = False)