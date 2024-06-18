import pandas as pd
from itertools import product

air_quality = pd.read_csv('data/airquality/air_quality.csv')

air_quality['DATA'] = pd.to_datetime(pd.to_datetime(air_quality['DATA'], format = '%d/%m/%Y').dt.date)
air_quality = air_quality.loc[air_quality.DATA.dt.year<2019]

# Convert the specified columns to numeric
columns = ['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h', '09h', '10h',\
            '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h', '19h', '20h',\
            '21h', '22h', '23h', '24h']
air_quality[columns] = air_quality[columns].apply(pd.to_numeric, errors='coerce')

# Create new columns min, max, mean
air_quality['min'] = air_quality[columns].min(axis=1)
air_quality['max'] = air_quality[columns].max(axis=1)
air_quality['mean'] = air_quality[columns].mean(axis=1)

# Extracting the hour of the column with the minimum value
air_quality['min_hour'] = air_quality[columns].idxmin(axis=1).str.extract(r'(\d+)')
air_quality['max_hour'] = air_quality[columns].idxmax(axis=1).str.extract(r'(\d+)')
air_quality['min_hour'] = pd.to_numeric(air_quality.min_hour)
air_quality['max_hour'] = pd.to_numeric(air_quality.max_hour)

######################################
######### AIR QUALITY LEGEND #########
######################################
air_quality_legend = air_quality[['NOM ESTACIO','CONTAMINANT','LATITUD', 'LONGITUD', 'LIMADM COMARCA']].drop_duplicates(subset = ['NOM ESTACIO','CONTAMINANT']).reset_index(drop = True)
air_quality_legend.columns = ['codi_estacio', 'contaminant', 'latitud', 'longitud', 'com_code']
air_quality_legend['com_code'] = air_quality_legend['com_code'].astype(str).str.zfill(2)
pc_muni_com_at_codes = pd.read_csv("data/geo_codes/pc_muni_com_at_codes.csv")
com_at_codes_legend = pc_muni_com_at_codes[['com_code','at_code']].drop_duplicates().reset_index(drop=True)
com_at_codes_legend["com_code"] = com_at_codes_legend["com_code"].astype(str).str.zfill(2)
com_at_codes_legend = com_at_codes_legend.sort_values('com_code').drop_duplicates(subset = 'com_code')
air_quality_legend = air_quality_legend.merge(com_at_codes_legend, on = 'com_code', how = 'inner')


######################################
######### AIR QUALITY TABLE ##########
######################################
dates = pd.date_range(start='2010-01-01', end='2018-12-31', freq='D')

# Create a DataFrame with all combinations of 'codi_estacio' and 'contaminant' for each date
date_combinations = pd.DataFrame(list(product(dates, air_quality_legend['codi_estacio'].unique(), air_quality_legend['contaminant'].unique())), columns=['date', 'codi_estacio', 'contaminant'])
date_combinations = pd.merge(date_combinations, air_quality_legend[['codi_estacio', 'contaminant']], on=['codi_estacio', 'contaminant'], how='inner')

# Rename columns in air_quality
air_quality.rename(columns={'NOM ESTACIO': 'codi_estacio',\
                            'CONTAMINANT': 'contaminant',\
                            'DATA': 'date'}, inplace=True)

merged_df = pd.merge(date_combinations, air_quality, how='left', on=['date', 'codi_estacio', 'contaminant'])
merged_df = merged_df[['codi_estacio','contaminant','date','min', 'max', 'mean','min_hour','max_hour'] + columns]
air_quality_at = air_quality.merge(air_quality_legend[['codi_estacio','at_code']].drop_duplicates(), on = 'codi_estacio', how = 'left')
air_quality_clean = air_quality_at[['date', 'contaminant','min', 'max', 'mean', 'min_hour', 'max_hour', 'at_code']]


######################################
######### AIR QUALITY AGG ############
######################################
air_quality_clean = air_quality_clean.groupby(['date', 'contaminant', 'at_code']).mean().reset_index()
air_quality_clean['na'] = air_quality_clean.isna().sum(axis=1)
air_quality_clean = air_quality_clean.loc[air_quality_clean.na<=0].reset_index(drop = True)
air_quality_clean['min_hour'] = air_quality_clean['min_hour'].round().astype(int)
air_quality_clean['max_hour'] = air_quality_clean['max_hour'].round().astype(int)
air_quality_clean = air_quality_clean.sort_values(['at_code','date']).reset_index(drop = True).drop(columns = 'na')

air_quality_clean.to_csv('data/airquality/air_quality_clean.csv', index = False)

######################################
######### AIR QUALITY PIVOT ##########
######################################
# Create pivot table
pivot_table = air_quality_clean.pivot_table(index=['date', 'at_code'], columns='contaminant', values=['min', 'max', 'mean', 'min_hour', 'max_hour'])

# Flatten the MultiIndex columns
pivot_table.columns = ['_'.join(col) for col in pivot_table.columns.values]

# Reset index
pivot_table = pivot_table.reset_index()


######################################
######### AIR QUALITY MERGE ##########
######################################
environment = pd.read_csv('../modelling_data/temp_hum_clean.csv')
environment['date'] = pd.to_datetime(environment['date'])
environment = environment.merge(pivot_table, on = ['at_code','date'], how = 'left')
environment.to_csv('../modelling_data/environment.csv', index = False)