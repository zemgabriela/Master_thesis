# =============== LIBRARIES ===============
import os
import requests
import pandas as pd
import numpy as np
import plotnine as p9
import geopandas as gpd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from zipfile import ZipFile
from itertools import product
from IPython.display import Image
from IPython.display import display
from mizani.breaks import date_breaks
from scipy.interpolate import interp1d
from mizani.formatters import percent_format, date_format

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Matplotlib settings
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
plt.rcParams['font.family'] = 'Georgia'
plt.rcParams['svg.fonttype'] = 'none'
set_matplotlib_formats('retina')
plt.rcParams['figure.dpi'] = 300

# Plotnine settings (for figures)

p9.options.set_option('base_family', 'Georgia')

p9.theme_set(
    p9.theme_bw()
    + p9.theme(panel_grid=p9.element_blank(),
               legend_background=p9.element_blank(),
               panel_grid_major=p9.element_line(size=.5, linetype='dashed',
                                                alpha=.15, color='black'),
               plot_title=p9.element_text(ha='center'),
               dpi=300,
               panel_background=p9.element_rect(alpha=1, color='gray'),
               plot_background=p9.element_rect(alpha=0),
    )
)

# =============== GENERAL MAP ===============
# =============== READ FILES ===============
pc_muni_com_at_df = pd.read_csv('data/geo_codes/pc_muni_com_at_codes.csv')

muni_shapes = (gpd.read_file('data/shapefiles/municipis/divisions-administratives-v2r1-municipis-50000-20230707.shp')
    .rename(columns={'CODIMUNI': 'muni_code'})
    [['muni_code', 'geometry']]
)

# =============== MERGE TO GET COORDINATES ===============
muni_com_df = (pc_muni_com_at_df
               [['muni_code', 'muni_name', 'com_code', 'com_name']]
               .drop_duplicates()
               .assign(muni_code=lambda dd: dd.muni_code.astype(str).str.zfill(6))
)

at_muni_df = (pc_muni_com_at_df
               [['muni_code', 'muni_name', 'at_code', 'at_name']]
               .drop_duplicates()
               .assign(muni_code=lambda dd: dd.muni_code.astype(str).str.zfill(6)))

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

at_shapes['legend'] = at_shapes['at_code'] + ':' + at_shapes['at_name']

# =============== MAP ===============
#| code-fold: true
# (p9.ggplot(muni_shapes)
#  + p9.geom_map(data=at_shapes, alpha=.8, size=.4, mapping=p9.aes(fill='legend'))
#  + p9.geom_map(alpha=0, size=.05)
#  + p9.geom_map(data=com_shapes, alpha=0, size=.25)
#  + p9.theme_void()
#  + p9.labs(fill='')
#  + p9.guides(fill=p9.guide_legend(ncol=2))
#  + p9.theme(figure_size=(6, 6),
#             dpi=300,
#             legend_position=(.65, .175))
#  ).draw()

# =============== CARDIO - AGE, SEX===============
# =============== READ FILES ===============
cardio_clean = pd.read_csv('data/AMI/cardio_cat_clean.csv')
pop_cat_ts = pd.read_csv('data/population/clean/catalunya/5Y.csv')

# =============== AMI ALERTS HISTOGRAM - SEX, AGE ===============
# (p9.ggplot(cardio_clean.groupby(['age', 'sex']).size().rename('events').reset_index()
#            .assign(events=lambda dd: np.where(dd['sex'] == 'Male', dd.events, dd.events * -1)),
#            p9.aes('age', 'events', fill='sex'))
#  + p9.geom_col()
#  + p9.coord_flip()
#  + p9.labs(x='Age (years)', y='AMI alerts', fill='',
#            title='Age and sex distribution of all AMI alerts (2010-2018)')
#  + p9.scale_fill_manual(values=["#ED553B", "#3CAEA3"])  # Specify your color palette here
#  + p9.theme(legend_position=(.8, .9),
#             title=p9.element_text(ha='center', size=10))
# )

# =============== AMI ALERTS DEVELOPMENT (SMOOTHED) ===============
cardio_clean.date =pd.to_datetime(cardio_clean['date'], format='%Y-%m-%d')
colors = ['black', '#b30000', '#0d88e6', 'orange']
labels = ['Daily', '7 days MA', '28 days MA', '365 days MA']

# (cardio_clean
#  .set_index('date')
#  .assign(events=1)
#  .resample('D')
#  .events
#  .sum()
#  .reset_index()
#  .assign(f_rolling_week=lambda dd: 
#          dd.events.rolling(center=True, window=7).mean())
#  .assign(g_rolling_month=lambda dd:
#         dd.events.rolling(center=True, window=28).mean())
#  .assign(h_rolling_year=lambda dd: 
#         dd.events.rolling(center=True, window=365).mean())
#  .melt('date')
#  .dropna()
#  .pipe(lambda dd: p9.ggplot(dd) 
# + p9.aes('date', 'value', color='variable', alpha='variable', size='variable') 
# + p9.geom_line()
# + p9.scale_color_manual(colors, labels=labels)
# + p9.scale_size_manual([.4, .6, .6, .8], labels=labels)
# + p9.scale_x_datetime(breaks=date_breaks('2 years'), expand=(.01, .01))
# + p9.scale_alpha_manual([.3, 1, 1, 1], labels=labels)
# + p9.labs(x='', y='AMI alerts', color='', size='', alpha='', title='')
# + p9.ylim(0, 15)
# + p9.theme(figure_size=(6, 3),
#            legend_position='top',
#                 )
#  )
# ).draw()

# =============== CONVERT AGE TO CATHEGORICAL ===============
age_ranges = [
    '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
    '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
    '70-74', '75-79', '80-84', '85-89', '90-94', '95+']

pop_cat_ts = (pop_cat_ts
 .replace({'age_range':{'95-99': '95+', '100+': '95+'}})
 .assign(age_range=lambda dd: 
         pd.Categorical(dd.age_range, categories=age_ranges, ordered=True))
 .groupby(['year', 'age_range', 'sex'], as_index=False)
 ['population'].mean()
)

# =============== YEARLY AMI ALERTS - SEX, AGE ===============
cardio_yearly = (cardio_clean
      .assign(events=1)
      .assign(year=lambda dd: dd.date.dt.year)
      .groupby(['year', 'age_range', 'sex'])
      ['events']
      .sum()
      .reset_index()
      .assign(age_range=lambda dd: 
         pd.Categorical(dd.age_range, categories=age_ranges, ordered=True))
)

pop_cat_ts['abs_population'] = pop_cat_ts['population'].abs()
max_population = pop_cat_ts['abs_population'].max()


# for idx, year in enumerate(range(2010, 2019)):
#     display(cardio_yearly
#      .query(f'year=={year}')
#      .assign(events=lambda dd: np.where(dd.sex=='Male', dd.events, dd.events * -1))
#      .pipe(lambda dd: p9.ggplot(dd)
#     + p9.aes('age_range', 'events', fill='sex')
#     + p9.geom_col()
#     + p9.coord_flip()
#     + p9.labs(fill='', x='', y='', title=f'AMI alerts {year}')
#     + p9.scale_y_continuous(
#         breaks=[-300, -150, 0, 150, 300],
#         limits=(-cardio_yearly.events.max(), cardio_yearly.events.max()),
#         labels=['300', '150', '0', '150', '300']
#                  )
#     + p9.theme(figure_size=(3.5, 5),
#     legend_position=(.35, .925),
#     panel_grid_major_y=p9.element_blank(),
#     dpi=600,
#     plot_background=
#     p9.element_rect(fill='#eee8d5', color='#eee8d5', alpha=1),
#                               )
#            )
#     )

# =============== YEARLY POPULATION - SEX, AGE ===============
# # Create subplots
# fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex = True, sharey = True)

# # Define the colors
# male_color = "#3CAEA3"
# female_color = "#ED553B"

# # Function to plot population pyramid for a given year
# def plot_population_pyramid(ax, data, year, max_population):
#     male_data = data[(data['year'] == year) & (data['sex'] == 'Male')]
#     female_data = data[(data['year'] == year) & (data['sex'] == 'Female')]
    
#     # Ensure populations are negative for females for plotting
#     female_data['population'] = -female_data['population']
    
#     # Plot males and females
#     ax.barh(male_data['age_range'], male_data['population'], color=male_color, label='Male')
#     ax.barh(female_data['age_range'], female_data['population'], color=female_color, label='Female')
    
#     # Customize the plot
#     ax.set_xlim(-max_population, max_population)
#     ax.set_xticks([-250000, -125000, 0, 125000, 250000])
#     ax.set_xticklabels(['250k', '125k', '0', '125k', '250k'])
#     ax.set_title(f"Population {year}", fontsize=12)
    
#     # Add legend only to the first subplot for clarity
#     if year == 2010:
#         ax.legend(loc='upper left')


# # Loop through each year and create a subplot
# for idx, year in enumerate(range(2010, 2019)):
#     ax = axs.flat[idx]
#     plot_population_pyramid(ax, pop_cat_ts, year, max_population)

# # Adjust layout
# plt.tight_layout()
# # Save the figure as PDF
# plt.savefig('../plots/population_pyramids.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# =============== YEARLY INCIDENCE AMI - SEX, AGE ===============
incidence_yearly = (cardio_yearly
    .merge(pop_cat_ts)
    .query('age_range>="20-24"')
    .assign(incidence=lambda dd: dd.events / dd.population * 100_000)
    .assign(age_range=lambda dd: pd.Categorical(dd.age_range, categories=age_ranges[4:], ordered=True))
)

# # Create subplots
# fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex = True, sharey = True)

# # Define the colors
# male_color = "#3CAEA3"
# female_color = "#ED553B"

# # Function to plot AMI incidence for a given year
# def plot_ami_incidence(ax, data, year):
#     male_data = data[(data['year'] == year) & (data['sex'] == 'Male')]
#     female_data = data[(data['year'] == year) & (data['sex'] == 'Female')]
    
#     # Ensure incidences are negative for females for plotting
#     female_data['incidence'] = -female_data['incidence']
    
#     # Plot males and females
#     ax.barh(male_data['age_range'], male_data['incidence'], color=male_color, label='Male')
#     ax.barh(female_data['age_range'], female_data['incidence'], color=female_color, label='Female')
    
#     # Customize the plot
#     ax.set_xticks([-150, -75, 0, 75, 150])
#     ax.set_xticklabels(['150', '75', '0', '75', '150'])
#     ax.set_title(f"AMI Incidence {year}", fontsize=12)
#     ax.set_xlabel('')
#     ax.set_ylabel('')

#     # Add legend only to the first subplot for clarity
#     if year == 2010:
#         ax.legend(loc='upper left')

# # Loop through each year and create a subplot
# for idx, year in enumerate(range(2010, 2019)):
#     ax = axs.flat[idx]
#     plot_ami_incidence(ax, incidence_yearly, year)

# # Adjust layout
# plt.tight_layout()

# # Save the figure as PDF
# plt.savefig('../plots/ami_incidence.pdf', dpi=300, bbox_inches='tight')

# # Show the figure
# plt.show()

# =============== AMI INCIDENCE ESTIMATES - SEX, AGE ===============
incidence_estimates = (
    incidence_yearly
    .groupby(['age_range', 'sex'], as_index=False)
    .incidence
    .mean()
)

# (incidence_estimates
#  .assign(incidence=lambda dd: 
#  np.where(dd.sex=='Male', dd.incidence, dd.incidence * -1))
#  .assign(label_y=lambda dd: 
#  np.where(dd.sex=='Male', dd.incidence + 15, dd.incidence - 15))
#  .pipe(lambda dd: p9.ggplot(dd)
#  + p9.aes('age_range', 'incidence', fill='sex')
#     + p9.geom_col()
#     + p9.coord_flip()
#     + p9.guides(fill=False)
#     + p9.labs(fill='', x='', y='AMI alerts per 100k',
#               title='Age and sex AMI incidence estimates',
#     )    
#     + p9.geom_text(p9.aes(label='incidence.round(1).abs()', y='label_y'),
#                    va='center', size=7,)
#                    + p9.scale_fill_manual(values=["#ED553B", "#3CAEA3"])  # Specify your color palette here
#     + p9.scale_y_continuous(
#         breaks=[-150, -75, 0, 75, 150],
#         limits=(-incidence_yearly.incidence.max(),
#                 incidence_yearly.incidence.max()),
#         labels=['150', '75', '0', '75', '150']
#                  )
#     + p9.scale_x_discrete(breaks=age_ranges[4:])
#     + p9.theme(figure_size=(4, 5), panel_grid_major_y=p9.element_blank()
#         )
#  )
# ).draw()

# =============== AMI INCIDENCE EXPECTED/OBSERVED TREND - SEX, AGE ===============
(incidence_estimates
 .merge(pop_cat_ts.query('2010 <= year <= 2018'), how='left'))

(incidence_estimates
 .merge(pop_cat_ts.query('2010 <= year <= 2018'), how='left')
 .assign(expected=lambda dd: dd.incidence * dd.population / 100_000)
 .merge(cardio_yearly, how='left')
 .groupby(['year'], as_index=False)
 .agg({'expected': 'sum', 'events': 'sum'})
 .assign(expected_rate=lambda dd: dd.expected / dd.expected.iloc[0])
 .assign(observed_rate=lambda dd: dd.events / dd.events.iloc[0])
)

incidence_trend_estimates = (incidence_estimates
 .merge(pop_cat_ts.query('2010 <= year <= 2018'), how='left')
 .assign(expected=lambda dd: dd.incidence * dd.population / 100_000)
 .merge(cardio_yearly, how='left')
 .groupby(['year'], as_index=False)
 .agg({'expected': 'sum', 'events': 'sum'})
 .assign(expected_rate=lambda dd: dd.expected / dd.expected.iloc[0])
 .assign(observed_rate=lambda dd: dd.events / dd.events.iloc[0])
)

# (incidence_trend_estimates
#  .melt('year')
#  .query('not variable.str.contains("rate")')
#   .replace({'variable': {'events':  'Observed', 'expected': 'Expected*'}})
#  .pipe(lambda dd: p9.ggplot(dd) 
#     + p9.aes('year', 'value', color='variable')
#     + p9.geom_line()
#     + p9.geom_point()
#     + p9.labs(x='', y='AMI Alerts', color='', 
#               caption='*Based on population estimates and incidence 2010-2018')
#     + p9.theme(
#         figure_size=(5, 2.5),
#         legend_position=(.425, .9),
    
#     )
#  )
# ).draw()

# =============== AMI INCIDENCE RELATIVE INCREASE - SEX, AGE ===============
# (incidence_trend_estimates
#  .melt('year')
#  .query('variable.isin(["expected_rate", "observed_rate"])')
#  .replace({'variable': {
#     'expected_rate': 'Expected AMI alerts',
#     'observed_rate': 'Observed AMI alerts'}})
#  .pipe(lambda dd: p9.ggplot(dd) 
#     + p9.aes('year', 'value', color='variable')
#     + p9.geom_line()
#     + p9.geom_point()
#     + p9.labs(x='', y='Relative increase', color='')
#     + p9.theme(
#         figure_size=(5, 2.5),
#         legend_position=(.425, .9),
    
#     )
#  )
# ).draw()

# =============== AGE,SEX SPECIFIC INCIDENCE RATE ===============
# display(incidence_yearly
#  .query('"30-34" <= age_range <= "85-89"')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.aes('year', 'incidence', color='sex')
#   + p9.geom_line()
#   + p9.geom_point(size=.3)
#   + p9.facet_wrap('age_range', ncol=4, labeller=lambda x: x + " years old")
#   + p9.labs(x='', y='AMI alerts per 100k', color='')
#   + p9.scale_y_continuous(limits=(0, None))
#   + p9.scale_x_continuous(breaks=range(2011, 2019, 2))
#   + p9.theme(
#     figure_size=(6, 4.5),
#     axis_text_x=p9.element_text(size=7),
#     legend_position='top',
#     legend_key_size=7,
#     legend_text=p9.element_text(size=8)             
#     )
#  )
#  )

# display(incidence_yearly
#  .query('"30-34" <= age_range <= "85-89"')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.aes('year', 'incidence', color='sex')
#   + p9.geom_smooth(method='lm', se=True, size=.5, linetype='dashed')
# #   + p9.geom_line()
#   + p9.geom_point(size=.3)
#   + p9.facet_wrap('age_range', ncol=4, labeller=lambda x: x + " years old")
#   + p9.labs(x='', y='AMI alerts per 100k', color='')
#   + p9.scale_y_continuous(limits=(0, None))
#   + p9.scale_x_continuous(breaks=range(2011, 2019, 2))
#   + p9.theme(
#     figure_size=(6, 4.5),
#     axis_text_x=p9.element_text(size=7),
#     legend_position='top',
#     legend_key_size=7,
#     legend_text=p9.element_text(size=8)             
#     )
#  )
#  )

def fit_linear_model(df, x, y):
    X = sm.add_constant(df[x])  # Adding a constant term to the predictor
    y = df[y]

    model = sm.OLS(y, X)
    results = model.fit()

    # Get slope (coefficient for x) and its confidence interval
    slope = results.params[x]
    slope_conf_int = results.conf_int(alpha=0.05).loc[x]

    # Get R-squared and adjusted R-squared
    r2_score = results.rsquared

    # Get p-value and standard error for the slope
    slope_p_value = results.pvalues[x]
    slope_std_err = results.bse[x]

    return pd.Series({
        'trend': slope,
        'ci_low': slope_conf_int[0],
        'ci_high': slope_conf_int[1],
        'r2_score': r2_score,
        'p_value': slope_p_value,
        'SE': slope_std_err
    })

# =============== AGE,SEX SPECIFIC INCIDENCE RATE LIN. REG. TABLE ===============
age_trend_results = (incidence_yearly
 .groupby(['age_range', 'sex'], as_index=False)
 .apply(fit_linear_model, 'year', 'incidence')
 .rename(columns={'age_range': 'Age', 'sex': 'Sex', 'r2_score': 'R2',
                  'trend': 'Trend', 'ci_low': 'T (0.025)',
                  'ci_high': 'T (0.975)', 'p_value': 'p-value'})
)

# (age_trend_results
#  .style
#  .hide(axis='index')
#  .format('{:.3f}', subset=['Trend', 'T (0.025)', 'T (0.975)', 'R2', 'p-value', 'SE'])
#  .apply(lambda x: ['color: black' if x['p-value'] < 0.05 else 'color: gray' for i in x], axis=1)
# )

# =============== CARDIO - SPATIAL ===============

# =============== CARDIO - COMARQUE ===============
# =============== READ FILES ===============
pop_com_ts = (pd.read_csv('data/population/clean/comarques/5Y.csv')
              .assign(com_code=lambda dd: dd['com_code'].astype(str).str.zfill(2))
              .rename(columns={'codi_comarca': 'com_code'}))

come_code_names = (
    muni_com_df
    [['com_code', 'com_name']]
    .drop_duplicates()
    .assign(com_code=lambda dd: dd.com_code.astype(str).str.zfill(2))
)

pop_com_yearly = (pop_com_ts
    .groupby(['year', 'com_code', 'age_range', 'sex'], as_index=False)
    ['population']
    .mean()
    .groupby(['year', 'com_code'], as_index=False)
    ['population']
    .sum()
)

# =============== INCIDENCE PER 100K INHABITANTS ===============
coms_yearly_incidence = (cardio_clean
 .assign(year=lambda dd: dd.date.dt.year)
 .assign(age_range=lambda dd: dd.age_range.astype(str))
 .assign(com_name=lambda dd: pd.Categorical(dd.com_name))
 .groupby(['year', 'com_name'], as_index=False)
 .size()
 .rename(columns={'size': 'alerts'})
 .merge(come_code_names)
 .merge(pop_com_yearly, how='left', on=['com_code', 'year'])
 .assign(incidence=lambda dd: dd.alerts / dd.population * 100_000)
)

# display(coms_yearly_incidence
#  .merge(com_shapes, how='left')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.geom_map(data=com_shapes, fill='gray', size=.1)
#   + p9.geom_map(p9.aes(fill='incidence'), size=.1)
#   + p9.scale_fill_continuous('Oranges', limits=(0, 85))
#   + p9.facet_wrap('~year', ncol=3)
#   + p9.guides(fill=p9.guide_colorbar(
#       title='AMI alerts per 100k', 
#       barwidth=10,
#       barheight=16))
#   + p9.theme_void()
#   + p9.theme(
#     figure_size=(7, 8),
#     dpi=300,
#     strip_text=p9.element_text(size=11),
#     legend_position='bottom',
#     legend_text=p9.element_text(size=8, va='bottom', ha='center'),
#     legend_title=p9.element_text(ha='center', x=50),
#   )
# )
# )

# =============== ASIR ===============
com_yearly_incidences_std = (pop_com_ts
 .groupby(['com_code', 'year', 'sex', 'age_range'], as_index=False)
 ['population'].mean()
.query('2010 <= year <= 2018')
 .merge(incidence_estimates, how='inner')
 .assign(expected=lambda dd: dd.population * dd.incidence / 100_000)
 .groupby(['com_code', 'year'], as_index=False)
 ['expected']
 .sum()
 .merge(pop_com_yearly)
 .eval('expected_incidence = expected / population * 100_000')
 .eval('correction_factor = expected_incidence / expected_incidence.mean()')
 .merge(coms_yearly_incidence, how='left')
 .eval('asir = incidence / expected_incidence')
)

# display(com_yearly_incidences_std
#  .merge(com_shapes, how='left')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.geom_map(data=com_shapes, fill='gray', size=.1)
#   + p9.geom_map(p9.aes(fill='asir'), size=.1)
#   + p9.scale_fill_continuous('RdYlGn_r', limits=(.4, 1.6))
#   + p9.facet_wrap('~year', ncol=3)
#   + p9.guides(fill=p9.guide_colorbar(
#       title='Age-standardized incidence rate', 
#       barwidth=12,
#       barheight=24))
#   + p9.theme_void()
#   + p9.theme(
#     figure_size=(7, 8),
#     dpi=300,
#     strip_text=p9.element_text(size=11),
#     legend_position='bottom',
#     legend_text=p9.element_text(size=8, va='bottom', ha='left'),
#     legend_title=p9.element_text(x=100, y=30),
#   )
# )
# )

# =============== CORRECTION FACTOR ===============
# display(com_yearly_incidences_std
#  .merge(com_shapes, how='left')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.geom_map(data=com_shapes, fill='gray', size=.1)
#   + p9.geom_map(p9.aes(fill='correction_factor'), size=.1)
#   + p9.scale_fill_continuous('RdYlGn_r', limits=(0.75, 1.25))
#   + p9.facet_wrap('~year', ncol=3)
#   + p9.guides(fill=p9.guide_colorbar(
#       title='Age-corrected population risk factor', 
#       barwidth=12,
#       barheight=24))
#   + p9.theme_void()
#   + p9.theme(
#     figure_size=(7, 8),
#     dpi=300,
#     strip_text=p9.element_text(size=11),
#     legend_position='bottom',
#     legend_text=p9.element_text(size=8, va='bottom', ha='left'),
#     legend_title=p9.element_text(x=100, y=30),
#   )
# )
# )

# =============== EXPECTED INCIDENCE ===============
# display(pop_com_ts
#  .groupby(['com_code', 'year', 'sex', 'age_range'], as_index=False)
#  ['population'].mean()
# .query('2010 <= year <= 2018')
#  .merge(incidence_estimates, how='inner')
#  .assign(expected=lambda dd: dd.population * dd.incidence / 100_000)
#  .groupby(['com_code', 'year'], as_index=False)
#  ['expected']
#  .sum()
#  .merge(pop_com_yearly)
#  .eval('expected_incidence = expected / population * 100_000')
#  .eval('correction_factor = expected_incidence / expected_incidence.mean()')
#  .merge(com_shapes, how='left')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.geom_map(data=com_shapes, fill='gray', size=.1)
#   + p9.geom_map(p9.aes(fill='expected_incidence'), size=.1)
#   + p9.scale_fill_continuous('Oranges', limits=(0, 85))
#   + p9.facet_wrap('~year', ncol=3)
#   + p9.guides(fill=p9.guide_colorbar(
#       title='Expected AMI alerts per 100k', 
#       barwidth=12,
#       barheight=24))
#   + p9.theme_void()
#   + p9.theme(
#     figure_size=(7, 8),
#     dpi=300,
#     strip_text=p9.element_text(size=11),
#     legend_position='bottom',
#     legend_text=p9.element_text(size=8, va='bottom', ha='left'),
#     legend_title=p9.element_text(x=100, y=30),
#   )
# )
# )

# =============== CARDIO - AMBITS TERRITORIALS ===============
# =============== READ FILES ===============
pop_at_ts = (pd.read_csv('data/population/clean/ambits_territorials/5Y.csv')
             .assign(month=lambda dd: dd['month'].astype(str).str.zfill(2)))

at_code_names = at_muni_df[['at_code', 'at_name']].drop_duplicates()

# =============== EXTRAPOLATED POOPULATION IN PENEDES ===============
year_months = pd.DataFrame(list(product(
        range(2010, 2019),
        ['01', '07'],
        age_ranges,
        ['Male', 'Female'],
        at_code_names['at_code'].unique())),
        columns=['year', 'month', 'age_range', 'sex', 'at_code'])

def extrapolate_linear(s):
    s = s.copy()
    # Indices of not-nan values
    idx_nn = s.index[~s.isna()]
    
    # At least two data points needed for trend analysis
    assert len(idx_nn) >= 2
    
    # Outermost indices
    idx_l = idx_nn[0]
    idx_r = idx_nn[-1]
    
    # Indices left and right of outermost values
    idx_ll = s.index[s.index < idx_l]
    idx_rr = s.index[s.index > idx_r]
    
    # Derivative of not-nan indices / values
    v = s[idx_nn].diff()
    
    # Left- and right-most derivative values
    v_l = v.iloc[1]
    v_r = v.iloc[-1]
    f_l = idx_l - idx_nn[1]
    f_r = idx_nn[-2] - idx_r
    
    # Set values left / right of boundaries
    l_l = lambda idx: (idx_l - idx) / f_l * v_l + s[idx_l]
    l_r = lambda idx: (idx_r - idx) / f_r * v_r + s[idx_r]
    x_l = pd.Series(idx_ll).apply(l_l)
    x_l.index = idx_ll
    x_r = pd.Series(idx_rr).apply(l_r)
    x_r.index = idx_rr
    s[idx_ll] = x_l
    s[idx_rr] = x_r
    return s


extrapolated_pop_at_ts = (pop_at_ts
.assign(ym=lambda dd: dd.year.astype(str) + dd.month)
.query('ym >= "201107"')
.query('year <= 2018')
 .merge(year_months, how='right')
 .assign(date=lambda dd: pd.to_datetime(dd.year.astype('str') 
            + dd.month, format='%Y%m'))
 .assign(extrapolated=lambda dd: np.where(dd.population.isna(),
                                     'Extrapolated', 'Observed'))
 .groupby(['at_code', 'age_range', 'sex'], as_index=False)
 .apply(lambda dd: dd.assign(population=lambda dd: 
    extrapolate_linear(dd['population'])))
)

# display(extrapolated_pop_at_ts
#  .query('at_code=="AT08"')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.aes('date', 'population', fill='sex', color='extrapolated', group='sex')
#   + p9.geom_line()
#   + p9.scale_x_datetime(breaks=date_breaks('3 years'), expand=(.01, .01),
#                         labels=date_format('%Y'))
#   + p9.facet_wrap('~age_range')
#   + p9.labs(x='', y='Total Population', color='', 
#             title='Population estimates for AT08 - Penedès')
#   + p9.theme(
#     figure_size=(6, 4),
#     title=p9.element_text(ha='center', size=10),
#     axis_text=p9.element_text(size=7),
#   )
#  )
# )

# =============== AMI INCIDENCE RATE PER 100K INHABITANTS ===============
pop_at_yearly = (extrapolated_pop_at_ts
    .groupby(['year', 'at_code', 'age_range', 'sex'], as_index=False)
    ['population']
    .mean()
    .groupby(['year', 'at_code'], as_index=False)
    ['population']
    .sum()
)

at_yearly_incidence = (cardio_clean
 .assign(year=lambda dd: dd.date.dt.year)
 .assign(age_range=lambda dd: dd.age_range.astype(str))
 .groupby(['year', 'at_code'], as_index=False)
 .size()
 .rename(columns={'size': 'alerts'})
 .merge(pop_at_yearly, how='left', on=['at_code', 'year'])
 .assign(incidence=lambda dd: dd.alerts / dd.population * 100_000)
)

# display(at_yearly_incidence
#  .merge(at_shapes, how='left')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.geom_map(data=com_shapes, fill='gray', size=.1)
#   + p9.geom_map(p9.aes(fill='incidence'), size=.1)
#   + p9.scale_fill_continuous('Oranges')
#   + p9.facet_wrap('~year', ncol=3)
#   + p9.guides(fill=p9.guide_colorbar(
#       title='AMI alerts per 100k', 
#       barwidth=10,
#       barheight=16))
#   + p9.theme_void()
#   + p9.theme(
#     figure_size=(7, 8),
#     dpi=300,
#     strip_text=p9.element_text(size=11),
#     legend_position='bottom',
#     legend_text=p9.element_text(size=8, va='bottom', ha='center'),
#     legend_title=p9.element_text(ha='center', x=50),
#   )
# )
# )

# =============== POPULATION PYRAMID IN 2018 - SEX, AGE, AT ===============
# display(extrapolated_pop_at_ts
#  .query('year==2018')
#  .groupby('at_code', as_index=False)
#  .apply(lambda dd: dd.eval('pop_fraction = population / population.sum()')
#  .assign(pop_fraction=lambda dd: np.where(dd.sex=='Male', dd.pop_fraction, dd.pop_fraction * -1)))
#  .assign(age_range=lambda dd: pd.Categorical(dd.age_range, categories=age_ranges, ordered=True))
#  .merge(at_code_names)
#  .pipe(lambda dd: 
#     p9.ggplot(dd)
#     + p9.aes('age_range', 'pop_fraction', fill='sex')
#     + p9.geom_col()
#     + p9.coord_flip()
#     + p9.facet_wrap('at_name', ncol=4)
#     + p9.scale_y_continuous(labels=lambda x: np.round(np.abs(x), 3),
#                             limits=(-.05, .05))
#     + p9.guides(fill=False)
#     + p9.labs(x='', y='', title='Population Pyramid for Catalan ATs (2018)')
#     + p9.theme(figure_size=(8, 5), 
#                axis_text_y=p9.element_text(size=7),
#                title=p9.element_text(ha='center', size=10))
#     )
#  )

# =============== ASIR DEVELOPMENT OVER YEARS PER AT ===============
at_yearly_incidences_std = (extrapolated_pop_at_ts
 .groupby(['at_code', 'year', 'sex', 'age_range'], as_index=False)
 ['population'].mean()
.query('2010 <= year <= 2018')
 .merge(incidence_estimates, how='inner')
 .assign(expected=lambda dd: dd.population * dd.incidence / 100_000)
 .groupby(['at_code', 'year'], as_index=False)
 ['expected']
 .sum()
 .merge(pop_at_yearly)
 .eval('expected_incidence = expected / population * 100_000')
 .eval('correction_factor = expected_incidence / expected_incidence.mean()')
 .merge(at_yearly_incidence, how='left')
 .eval('asir = incidence / expected_incidence')
)

# display(at_yearly_incidences_std
#  .merge(at_shapes, how='left')
#  .pipe(lambda dd: p9.ggplot(dd)
#   + p9.geom_map(p9.aes(fill='asir'), size=.1)
#   + p9.scale_fill_continuous('RdYlGn_r', limits=(.5, 1.5))
#   + p9.facet_wrap('~year', ncol=3)
#   + p9.guides(fill=p9.guide_colorbar(
#       title='Age-standardized incidence rate', 
#       barwidth=12,
#       barheight=24))
#   + p9.theme_void()
#   + p9.theme(
#     figure_size=(7, 8),
#     dpi=300,
#     strip_text=p9.element_text(size=11),
#     legend_position='bottom',
#     legend_text=p9.element_text(size=8, va='bottom', ha='left'),
#     legend_title=p9.element_text(x=100, y=30),
#   )
# )
# )

# =============== EXPECTED/OBSERVED PER AT ===============
# display(at_yearly_incidences_std
#  .groupby('at_code', as_index=False)
#  .apply(lambda dd: dd
#         .eval('expected_rate = expected / expected.iloc[0]')
#         .eval('observed_rate = incidence / incidence.iloc[0]')
#         )
#  .melt(id_vars=['at_code', 'year'])
#  .query('variable.str.contains("incidence")')
#  .merge(at_code_names)
#  .replace({'variable': {'expected_incidence': 'Expected Incidence',
#                         'incidence': 'Observed Incidence'}})
#  .pipe(lambda dd: 
#     p9.ggplot(dd) 
#   + p9.aes('year', 'value', color='variable')
#   + p9.geom_line()
#   + p9.geom_point(size=.7)
#   + p9.labs(x='', y='', color='')
#   + p9.scale_x_continuous(breaks=range(2011, 2019, 2))
#   + p9.facet_wrap('at_name', ncol=4)
#   + p9.theme(figure_size=(8, 4.5),
#              legend_position='top')
#   )
# )

# =============== AMI ALERTS BY AT AND YEAR ===============
# (cardio_clean
# .assign(year=lambda dd: dd.date.dt.year)
#  .groupby(['year', 'at_code', 'at_name'])
#  .size()
#  .rename('alerts')
#  .reset_index()
#  .pivot_table(index=['at_code', 'at_name'], columns='year', values='alerts')
# )

# =============== SMOOTHING AT DIFFERENT TEMPORAL SCALES ===============
dates_range = pd.DataFrame(dict(date=pd.date_range('2010-01-01', '2018-12-31', freq='D')))

daily_pop_at_ts = (extrapolated_pop_at_ts
 .groupby(['at_code', 'age_range', 'sex'], as_index=True)
 .apply(lambda dd: 
       dd.merge(dates_range, how='right')
        .set_index('date')
        .resample('D', )
        ['population']
        .median()
        .pipe(lambda x: extrapolate_linear(x))
)
.reset_index()
 .melt(['at_code', 'age_range', 'sex'])
 .groupby(['at_code', 'age_range', 'sex'], as_index=True)
 .apply(lambda dd: dd.assign(value=dd.value.interpolate(method='linear')))
 .reset_index(drop=True)
)

daily_pops_at = (daily_pop_at_ts
.groupby('at_code')
.apply(lambda dd: dd.set_index('date').resample('D')['value'].sum())
.reset_index()
.melt(['at_code'])
)

daily_pops_at['date'] = pd.to_datetime(daily_pops_at['date'], format = '%Y-%m-%d')

cardio_at_mas = (cardio_clean
 .groupby(['at_code'])
 .apply(lambda dd: dd
 .set_index('date')
 .assign(events=1)
 .resample('D')
 .events
 .sum()
 .reset_index()
 .assign(a_rolling_week=lambda dd: 
         dd.events.rolling(center=True, window=7).mean())
 .assign(b_rolling_month=lambda dd:
        dd.events.rolling(center=True, window=28).mean())
 .assign(c_rolling_trimester=lambda dd:
         dd.events.rolling(center=True, window=90).mean())
 .assign(d_rolling_year=lambda dd: 
        dd.events.rolling(center=True, window=365).mean())
 )
 .reset_index()
 .drop(columns='level_1')
 .merge(daily_pops_at, how='left')
 .rename(columns={'value': 'population'})
)


labels = ['7 days MA', '28 days MA', '90 days MA', '365 days MA']
# display(cardio_at_mas
#  .melt(['date', 'at_code', 'population'])
#  .dropna()
#  .assign(pop_value=lambda dd: dd.value / dd['population'] * 100_000)
#  .merge(at_code_names)
#  .query('variable.str.contains("rolling")')
#  .pipe(lambda dd: p9.ggplot(dd)
# + p9.aes('date', 'pop_value', color='variable', alpha='variable', size='variable') 
# + p9.geom_line()
# + p9.facet_wrap('at_code + ": " + at_name', ncol=4)
# + p9.scale_color_manual(colors, labels=labels)
# + p9.scale_size_manual([.2, .4, .6, .8], labels=labels)
# + p9.scale_alpha_manual([.2, .4, 1, 1], labels=labels)
# + p9.scale_x_datetime(breaks=date_breaks('2 years'),
#                       expand=(.01, .01),
#                       labels=date_format('%Y'))
# + p9.labs(x='', y='Daily AMI alerts per 100,000 inhabitants',
#           color='', size='', alpha='', title='')
# + p9.ylim(0, .6)
# + p9.theme(figure_size=(8, 5),
#            legend_position='top',
#            legend_text=p9.element_text(size=10),
#            legend_key_size=13,
#                 )
# )
# )

# =============== EXPECTED DAILY AMI ALERTS BASED ON POPULATION ESTIMATES ===============
expected_daily_ats = (at_yearly_incidences_std
 .assign(date=lambda dd: pd.to_datetime(dd.year.astype(str) + '-07'))
 .groupby('at_code')
 .apply(lambda dd: 
       dd.merge(dates_range, how='right')
        .set_index('date')
        .resample('D')
        [['expected']]
        .mean()
        .apply(lambda x: extrapolate_linear(x))
        .pipe(lambda x: x / 365)
)
.reset_index()
 .groupby(['at_code'], as_index=True)
 .apply(lambda dd: dd.assign(expected=dd['expected'].interpolate(method='linear')))
 .reset_index(drop=True)
 .merge(at_code_names)
)

# display(expected_daily_ats
#  .pipe(lambda dd: p9.ggplot(dd) 
#   + p9.aes('date', 'expected') 
#   + p9.geom_point(size=.1)
#   + p9.geom_point(
#       p9.aes(y='expected / 365'),
#       color='red',    
#       data=at_yearly_incidences_std
#       .assign(date=lambda dd: pd.to_datetime(dd.year.astype(str) + '-07'))
#       .merge(at_code_names)
#   )
#   + p9.facet_wrap('at_code + ": " + at_name', ncol=4, scales='free_y')
#   + p9.scale_x_datetime(
#   breaks=date_breaks('2 years'),
#   expand=(.01, .01),
#   labels=date_format('%Y'))
#   + p9.labs(x='', y='Expected daily AMI alerts', 
#             caption="*Based on population composition and age-specific risk")
#   + p9.theme(figure_size=(9, 3.5),
#         axis_text_y=p9.element_text(size=7),
#       )
#       )
# )

# =============== ASIR SMOOTHED ===============
# display(cardio_at_mas
#  .merge(expected_daily_ats)
#  .eval('a_asir_90=c_rolling_trimester / expected')
#  .eval('b_asir_365=d_rolling_year / expected')
#  .melt(['date', 'at_code', 'at_name'])
#  .query('variable.str.contains("asir")')
#   .pipe(lambda dd: 
#     p9.ggplot(dd)
#   + p9.aes('date', 'value', color='variable', size='variable', alpha='variable')
#   + p9.geom_line()
#   + p9.labs(x='', y='Daily ASIR', color='', size='', alpha='', title='')
#   + p9.facet_wrap('at_code + ": " + at_name', ncol=4)
#   + p9.ylim(0, 2)
#   + p9.scale_color_manual(['#0d88e6', 'orange'],
#       labels=labels[-2:])
#   + p9.scale_size_manual([.6, .8], labels=labels[-2:])
#   + p9.scale_alpha_manual([.6, 1], labels=labels[-2:])
#   + p9.scale_x_datetime(
#       breaks=date_breaks('2 years'),
#       expand=(.01, .01),
#       labels=date_format('%Y'))
#   + p9.theme(
#         figure_size=(8, 4.5),
#         legend_position='top',
#         legend_text=p9.element_text(size=12),
#         legend_key_size=15,
#   )
#   )
# )

# =============== SEASONALITY ===============
# display(cardio_at_mas
#  .merge(expected_daily_ats)
#  .eval('a_asir_90=c_rolling_trimester / expected')
#  .eval('b_asir_365=d_rolling_year / expected')
#  .eval('asir = events / expected')
#  .melt(['date', 'at_code', 'at_name'])
# #  .query('variable.str.contains("asir")')
#  .query('variable=="asir"')
#  .dropna()
#  .eval('day_of_year = date.dt.day_of_year')
#  .eval('year = date.dt.year')
# #  .query('variable.str.contains("90")')
#  .groupby(['at_code', 'at_name', 'year'], as_index=False)
#  .apply(lambda dd: dd.assign(value=lambda x: x.value / x.value.mean()))
#  .eval('month=date.dt.month')
#  .groupby(['at_code', 'at_name', 'month'], as_index=False)
#  ['value']
#  .agg(['mean', 'std'])
#  .pipe(lambda dd: p9.ggplot(dd) + p9.aes('month', 'mean - 1')
#        + p9.geom_col(p9.aes(fill='mean < 1'))
#        + p9.facet_wrap('at_code + ": " + at_name', ncol=4)
#        + p9.labs(x='', y='Relative ASIR', title='')
#        + p9.guides(fill=False)
#        + p9.theme(
#            figure_size=(8, 4.5),
#        )
# )
# )

# display(cardio_at_mas
#  .merge(expected_daily_ats)
#  .eval('a_asir_90=c_rolling_trimester / expected')
#  .eval('b_asir_365=d_rolling_year / expected')
#  .melt(['date', 'at_code', 'at_name'])
#  .query('variable.str.contains("asir")')
#  .dropna()
#  .eval('day_of_year = date.dt.day_of_year')
#  .eval('year = date.dt.year')
#  .query('variable.str.contains("90")')
#  .groupby(['at_code', 'at_name', 'year'])
#  .apply(lambda dd: dd.assign(value=lambda x: x.value / x.value.mean()))
#  .pipe(lambda dd: 
#     p9.ggplot(dd) 
#   + p9.aes('day_of_year', 'value') 
#   + p9.geom_line(p9.aes(group='year'), size=.3, alpha=.3)
#   + p9.stat_summary(fun_y=np.mean, geom='line', size=.8, color='black')
#   + p9.labs(x='', y='Yearly standardized ASIR')
#   + p9.scale_x_continuous(
#         breaks=[32, 91, 152, 213, 274, 335],
#         labels=['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'],
#         expand=(.01, 0))
#   + p9.facet_wrap('at_code + ": " + at_name', ncol=4)
#   + p9.ylim(.5, 1.5)
#   + p9.theme(
#         figure_size=(8, 4),
#         legend_position='top',
#         legend_text=p9.element_text(size=12),
#         legend_key_size=15,
#   )
#  )
# )


# =============== SAVE DATA ===============
cardio_at_asir = (cardio_at_mas.merge(expected_daily_ats)
                .eval('a_asir_90 = c_rolling_trimester / expected')
                .eval('b_asir_365 = d_rolling_year / expected')
                .eval('c_asir_5 = a_rolling_week / expected')
                .eval('b_asir_30 = b_rolling_month / expected'))

cardio_at_asir.to_csv('../modelling_data/cardio_at_asir.csv', index = False)