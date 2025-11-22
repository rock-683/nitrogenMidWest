#!/usr/bin/env python
# coding: utf-8

# # <font color=darkblue>Low Birthweight (LBW) Exploratory Data Analysis (EDA) & Visualization

# In[ ]:


import pandas as pd


# In[ ]:


# Read in file with low birthweight data

county_data = pd.read_csv('NCH2017.csv') # merged nitrogen_county and county_health tables
county_data.head()


# In[ ]:


# Data Dictionary
chr_datadictionary2017 = pd.read_csv('chr_dd2017.csv')

pd.set_option("max_rows", None)
pd.set_option('display.max_colwidth', 255)

chr_datadictionary2017['Release Year'] = chr_datadictionary2017['Release Year'].astype(str).apply(lambda x: x.replace('.0',''))

chr_datadictionary2017.iloc[[7],:]


# In[ ]:


#find range and central tendency of target health outcome
county_data['% Low Birthweight 20142020'].describe()


# In[ ]:


# Missouri Low Birth Rate Average - move to exploratory analysis for lbw

MLBW = county_data[county_data['STATE'] == 'MISSOURI']['% Low Birthweight 20142020'].sum()
ILBW = county_data[county_data['STATE'] == 'IOWA']['% Low Birthweight 20142020'].sum()
NLBW = county_data[county_data['STATE'] == 'NEBRASKA']['% Low Birthweight 20142020'].sum()
KLBW = county_data[county_data['STATE'] == 'KANASAS']['% Low Birthweight 20142020'].sum()

print('MLBW:',MLBW)
print(f"\nILBW: {ILBW}")
print(f"\nNLBW: {NLBW}")
print(f"\nKLBW: {KLBW}")

print(f"\nMWLBWavg: {(MLBW + ILBW + NLBW + KLBW)/4}")


# # Map of low birthweight 7 year average 2014 - 2020 by county
# 
# 2017 is center year

# In[ ]:


get_ipython().run_line_magic('pip', 'install -U plotly')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import plotly.figure_factory as ff

import matplotlib.pyplot as plt # Graphics
from matplotlib import colors
import seaborn # Graphics


# In[ ]:


# Load the county boundary coordinates
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[ ]:


# Build the choropleth
import plotly.express as px
fig = px.choropleth(county_data, 
    geojson=counties, 
    locations ='FIPS', 
    color='% Low Birthweight 20142020',
    color_continuous_scale= px.colors.sequential.PuBu,
    range_color=(0, 15),
    scope="usa"
    
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


fig.show()


# In[ ]:


county_data['STATE'].unique() 


# In[ ]:


# find % missing in Midwest (% Low Birthweight 20142020)

cd_lbw = county_data[['FIPS','STATE','CountyName','% Low Birthweight 20142020']]

print("Out of",cd_lbw.shape[0],"counties in the midwest,",cd_lbw.isna().sum()[3],
      "or", (round((cd_lbw.isna().sum())/(cd_lbw.shape[0])*100))[3], 
      "% are missing low birthweight (lbw) data.")

