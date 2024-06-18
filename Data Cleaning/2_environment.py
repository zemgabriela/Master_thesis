import pandas as pd
from sodapy import Socrata
import os
import plotnine as p9
from skimpy import skim
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# To suppress all warnings
warnings.filterwarnings("ignore")

# Extract legend using API
client = Socrata("analisi.transparenciacatalunya.cat", None)
# Metadata

# Metadades variables meteorològiques 
variables = client.get("4fb2-n3yi", limit=2000)
variables_legend = pd.DataFrame.from_records(variables).sort_values('codi_variable').reset_index(drop = True)

# Metadades estacions meteorològiques automàtiques
stations = client.get("yqwd-vj5e", limit=2000)
station_legend = pd.DataFrame.from_records(stations)

# translate 
variables_legend['nom_variable_esp'] = variables_legend['nom_variable']
# Define the translation dictionary
translation_dict = {
    'Pressió atmosfèrica màxima': 'Maximum Atmospheric Pressure',
    'Pressió atmosfèrica mínima': 'Minimum Atmospheric Pressure',
    'Humitat relativa màxima': 'Maximum Relative Humidity',
    'Velocitat del vent a 10 m (esc.)': 'Wind Speed at 10 m (scaled)',
    'Direcció de vent 10 m (m. 1)': 'Wind Direction at 10 m (m. 1)',
    'Temperatura': 'Temperature',
    'Humitat relativa': 'Relative Humidity',
    'Pressió atmosfèrica': 'Atmospheric Pressure',
    'Precipitació': 'Precipitation',
    'Irradiància solar global': 'Global Solar Irradiance',
    'Gruix de neu a terra': 'Snow Depth on Ground',
    'Temperatura màxima': 'Maximum Temperature',
    'Temperatura mínima': 'Minimum Temperature',
    'Humitat relativa mínima': 'Minimum Relative Humidity',
    'Velocitat del vent a 2 m (esc.)': 'Wind Speed at 2 m (scaled)',
    'Direcció del vent a 2 m (m. 1)': 'Wind Direction at 2 m (m. 1)',
    'Velocitat del vent a 6 m (esc.)': 'Wind Speed at 6 m (scaled)',
    'Direcció del vent a 6 m (m. 1)': 'Wind Direction at 6 m (m. 1)',
    'Ratxa màxima del vent a 10 m': 'Maximum Wind Gust at 10 m',
    'Direcció de la ratxa màxima del vent a 10 m': 'Direction of Maximum Wind Gust at 10 m',
    'Ratxa màxima del vent a 6 m': 'Maximum Wind Gust at 6 m',
    'Direcció de la ratxa màxima del vent a 6 m': 'Direction of Maximum Wind Gust at 6 m',
    'Ratxa màxima del vent a 2 m': 'Maximum Wind Gust at 2 m',
    'Direcció de la ratxa màxima del vent a 2 m': 'Direction of Maximum Wind Gust at 2 m',
    'Irradiància neta': 'Net Irradiance',
    'Precipitació màxima en 1 minut': 'Maximum Precipitation in 1 minute'
}

variables_legend['nom_variable'] = variables_legend['nom_variable_esp'].map(translation_dict)

import geopandas as gpd

# start by plotting our stations that contain values that will be analysed
station_legend['codi_comarca'] = station_legend['codi_comarca'].astype(str).str.zfill(2)

station_geometry = station_legend[['codi_estacio','latitud','longitud','codi_comarca']]
station_geometry = gpd.GeoDataFrame(station_geometry, geometry=gpd.points_from_xy(station_geometry['longitud'], station_geometry['latitud']))
station_geometry.latitud = pd.to_numeric(station_geometry.latitud)
station_geometry.longitud = pd.to_numeric(station_geometry.longitud)

pc_muni_com_at_df = (pd.read_csv('data/geo_codes/pc_muni_com_at_codes.csv')
                     .assign(com_code=lambda dd: dd.com_code.astype(str).str.zfill(2))
                     )
muni_com_df = (pc_muni_com_at_df
               [['muni_code', 'muni_name', 'com_code', 'com_name']]
               .drop_duplicates()
               .assign(muni_code=lambda dd: dd.muni_code.astype(str).str.zfill(6))
)

at_muni_df = (pc_muni_com_at_df
               [['muni_code', 'muni_name', 'at_code', 'at_name']]
               .drop_duplicates()
               .assign(muni_code=lambda dd: dd.muni_code.astype(str).str.zfill(6)))

muni_shapes = (gpd.read_file('data/shapefiles/municipis/divisions-administratives-v2r1-municipis-50000-20230707.shp')
    .rename(columns={'CODIMUNI': 'muni_code'})
    [['muni_code', 'geometry']]
)

com_shapes = (muni_shapes
 .merge(muni_com_df, validate='one_to_one')
 .assign(geometry=lambda dd: dd['geometry'].buffer(0.001))
 .dissolve(by='com_code')
 .reset_index()
 .assign(com_code=lambda dd: dd['com_code'].astype(str).str.zfill(2))
 [['com_code', 'com_name', 'geometry']]
)

at_shapes = (muni_shapes
 .merge(at_muni_df)
 .assign(geometry=lambda dd: dd['geometry'].buffer(0.001))
 .dissolve(by='at_code')
 .reset_index()
 [['at_code', 'at_name', 'geometry']]
)

# convert to the same crs
muni_shapes_latlong = muni_shapes.to_crs(epsg=4326)
at_shapes_latlong = at_shapes.to_crs(epsg=4326)
com_shapes_latlong = com_shapes.to_crs(epsg=4326)

com_shapes_latlong_with_points = com_shapes_latlong[com_shapes_latlong['com_code'].isin(station_geometry['codi_comarca'].dropna().unique())]

# display(p9.ggplot(muni_shapes_latlong) +
#  p9.geom_map(alpha=0.1, size=.1) +
#  p9.geom_map(data=com_shapes_latlong_with_points, alpha=0.4, size=.4, fill='lightgreen') +
#  p9.geom_point(data=station_geometry, mapping=p9.aes(x='longitud', y='latitud'), size = 1) +
#  p9.theme_void() +
#  p9.labs(fill='') +
#  p9.theme(figure_size=(6, 6),
#             dpi=300,
#             legend_position=(.65, .175)) +
# p9.ggtitle('Meteorological Stations by comarca lvl.')
#  )

muni_shapes_latlong_with_points = gpd.sjoin(muni_shapes_latlong, station_geometry, how="inner", op="intersects")

# display((p9.ggplot(muni_shapes_latlong) +
#  p9.geom_map(alpha=0.1, size=.1) +
#  p9.geom_map(data=muni_shapes_latlong_with_points, alpha=0.4, size=.4, fill='lightblue') +
#  p9.geom_point(data=station_geometry, mapping=p9.aes(x='longitud', y='latitud'), size = 1) +
#  p9.theme_void() +
#  p9.labs(fill='') +
#  p9.theme(figure_size=(6, 6),
#             dpi=300,
#             legend_position=(.65, .175)) +
# p9.ggtitle('Meteorological Stations grouped by municipality lvl.')
#  ).draw())
variables = ["42", "44", "40", "3"]
# Initialize an empty DataFrame to store the combined data
var_stat_data = pd.DataFrame()

for i in variables:
    folder_path = f"data/enironment/{i}"

    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through each file and read it into a DataFrame
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file is a CSV file (you can adjust the condition for other file types)
        if file_name.endswith('.csv'):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Append the DataFrame to the list
            dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_var_df = pd.concat(dfs, ignore_index=True)
    
    # Rename the 'value' column to the current variable
    combined_var_df.rename(columns={'VALOR_LECTURA': variables_legend.loc[variables_legend.codi_variable==i].reset_index(drop = True)["nom_variable"][0],\
                                     'CODI_ESTACIO' : 'codi_estacio',\
                                     'DATA_LECTURA': 'date'}, inplace=True)
    
    # Merge the current variable data with the combined data
    if var_stat_data.empty:
        var_stat_data = combined_var_df
    else:
        var_stat_data = pd.merge(var_stat_data, combined_var_df, on=['codi_estacio', 'date'], how='outer')

# del df, dfs, groupby_year, combined_var_df
# Convert 'date' column to datetime format
var_stat_data['date'] = pd.to_datetime(var_stat_data['date'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')

# Select columns for which you want to calculate daily averages
columns_to_aggregate = {
    'Maximum Relative Humidity': 'max',
    'Minimum Relative Humidity': 'min',
    'Maximum Temperature': 'max',
    'Minimum Temperature': 'min'
}

# Group by date and station, and calculate daily aggregate (maximum or minimum) for selected columns
var_stat_data['date'] = var_stat_data['date'].dt.strftime('%d/%m/%Y')
var_stat_data = var_stat_data.groupby(['date', 'codi_estacio']).agg(columns_to_aggregate).reset_index()

# Convert 'date' column back to datetime format
var_stat_data['date'] = pd.to_datetime(var_stat_data['date'], format='%d/%m/%Y')

# Extract year from the date
var_stat_data['year'] = var_stat_data['date'].dt.year
station_com_at = station_legend.rename(columns = {'codi_comarca':'com_code'})[['com_code','codi_estacio']]\
    .merge(pc_muni_com_at_df[['at_code','com_code']].drop_duplicates(), on = 'com_code')
var_stat_data_at_com = var_stat_data.merge(station_com_at, on = 'codi_estacio').drop_duplicates().sort_values(['codi_estacio','date']).reset_index(drop = True)
var_stat_data_at_com[['date','at_code']].drop_duplicates() #580018 
var_stat_data_at = var_stat_data_at_com.groupby(['date', 'at_code']).agg(columns_to_aggregate).reset_index()
var_stat_data_at['daily_diff_temp'] = abs(var_stat_data_at['Maximum Temperature'] - var_stat_data_at['Minimum Temperature'])
var_stat_data_at['daily_diff_hum'] = abs(var_stat_data_at['Maximum Relative Humidity'] - var_stat_data_at['Minimum Relative Humidity'])

columns_to_diff = ['Maximum Temperature', 'Minimum Temperature', 'Maximum Relative Humidity', 'Minimum Relative Humidity']

for column in columns_to_diff:
    var_stat_data_at[f'{column.lower().replace(" ", "_")}_change'] = (
        var_stat_data_at
        .sort_values(['at_code', 'date'])
        .reset_index(drop=True)
        .groupby('at_code')[column]
        .diff()
    )

# variable_name = ['Maximum Relative Humidity', 'Minimum Relative Humidity', 'Maximum Temperature', 'Minimum Temperature']
# for i in variable_name:
#     # ACF, PACF
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

#     # Plot ACF on the first subplot
#     acf_plot = plot_acf(var_stat_data_at[i], lags=100, ax=ax1, zero = False)
#     ax1.set_title(f'Autocorrelation Function \n (ACF) of {i}')
#     ax1.set_ylim([-0.25,0.25])
#     ax1.grid(True)
#     ax1.set_xlim([0, 100])

#     # Plot PACF on the second subplot
#     pacf_plot = plot_pacf(var_stat_data_at[i], lags = 50, ax=ax2, zero = False)
#     ax2.set_title(f'Partial Autocorrelation Function \n (PACF) of {i}')
#     ax2.set_ylim([-0.25,0.25])
#     ax2.grid(True)
#     ax2.set_xlim([0, 50])

#     # Adjust layout to prevent overlapping
#     plt.tight_layout()

#     # Show the plots
#     plt.show()

var_stat_data_at.to_csv('../modelling_data/temp_hum_clean.csv', index = False)