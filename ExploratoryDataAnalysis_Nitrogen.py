#!/usr/bin/env python
# coding: utf-8

# # Nitrogen Exploratory Data Analysis & Visualization

# In[2]:


import pandas as pd


# ### Reading in 1997- 2017 Nitrogen Data - Midwest
# 
# <font color=green>*County-level estimates of **kilograms** of nitrogen from Falcone (2021).*

# In[3]:


county_data = pd.read_csv('NCH2017.csv') # merged nitrogen_county and county_health tables
county_data.head()


# In[4]:


county_data.info()


# <font color=green>It looks like there are **411** counties in the nitrogen midwest dataset. All county-level estimates of kilograms of nitrogen are float64 or int64 data type.</font>

# In[5]:


#checking for missing values

pd.set_option("max_rows", None)
county_data.isna().sum()


# <font color=green> 17 counties are missing `Nitrogen Rate` data for the target year **2017**. Nitrogen Rate for years 1997, 2002, 2007 and 2012 all have at least 6 counties with missing data for `Nitrogen Rate`.

# In[6]:


nitrogen_df2 = county_data.iloc[0:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]] 
nitrogen_df2 = nitrogen_df2.dropna()
nitrogen_df2.head()


# In[10]:


# suppress scientific notation for all floats by adjust the pandas options.
pd.options.display.float_format = '{:.2f}'.format

nitrogen_df2.describe()


# <font color=green> For the target year, 2017, the Midwest range in nitrogen rate in a county is min:**36.29** kg /acre to max:**784.43** kg/acre. This does appear to be a rise in nitrogen rate from 1997, 2002, and 2007 amounts, but also a rise in the minimum amounts when compared to those same years. 

# In[7]:


NR2017 = nitrogen_df2['TotalNitrogen 2017'].sum() / nitrogen_df2['cropland 2017'].sum()
NR2012 = nitrogen_df2['TotalNitrogen 2012'].sum() / nitrogen_df2['cropland 2012'].sum()
NR2007 = nitrogen_df2['TotalNitrogen 2007'].sum() / nitrogen_df2['cropland 2007'].sum()
NR2002 = nitrogen_df2['TotalNitrogen 2002'].sum() / nitrogen_df2['cropland 2002'].sum()
NR1997 = nitrogen_df2['TotalNitrogen 1997'].sum() / nitrogen_df2['cropland 1997'].sum()

print('2017:',NR2017, ' 2012:',NR2012,  ' 2007:',NR2007, ' 2002:',NR2002, ' 1997:',NR1997)


# In[8]:


# Midwest Nitrogen Rate from 1997 to 2017 

MWNR1997 = nitrogen_df2['TotalNitrogen 1997'].sum() / nitrogen_df2['cropland 1997'].sum()
MWNR2002 = nitrogen_df2['TotalNitrogen 2002'].sum() / nitrogen_df2['cropland 2002'].sum()
MWNR2007 = nitrogen_df2['TotalNitrogen 2007'].sum() / nitrogen_df2['cropland 2007'].sum()
MWNR2012 = nitrogen_df2['TotalNitrogen 2012'].sum() / nitrogen_df2['cropland 2012'].sum()
MWNR2017 = nitrogen_df2['TotalNitrogen 2017'].sum() / nitrogen_df2['cropland 2017'].sum()


print('MWNR1997:',MWNR1997,  'MWNR2002:',MWNR2002, 'MWNR2007:', MWNR2007, 'MWNR2012:', MWNR2012, 'MWNR2017:', MWNR2017)
print(f"\nMWNRavg: {(MWNR1997 + MWNR2002 + MWNR2007 + MWNR2012 + MWNR2017)/5}")


# In[16]:


# Missouri Nitrogen Rate from 1997 to 2017

MONR1997 = nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['TotalNitrogen 1997'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['cropland 1997'].sum()
MONR2002 = nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['TotalNitrogen 2002'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['cropland 2002'].sum()
MONR2007 = nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['TotalNitrogen 2007'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['cropland 2007'].sum()
MONR2012 = nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['TotalNitrogen 2012'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['cropland 2012'].sum()
MONR2017 = nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['TotalNitrogen 2017'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'MISSOURI']['cropland 2017'].sum()

print('MONR1997:',MONR1997,  'MONR2002:',MONR2002, 'MONR2007:', MONR2007, 'MONR2012:', MONR2012, 'MONR2017:', MONR2017)
print(f"\nMWNRavg: {(MONR1997 + MONR2002 + MONR2007 + MONR2012 + MONR2017)/5}")


# In[9]:


# Kansas Nitrogen Rate from 1997 to 2017

KSNR1997 = nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['TotalNitrogen 1997'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['cropland 1997'].sum()
KSNR2002 = nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['TotalNitrogen 2002'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['cropland 2002'].sum()
KSNR2007 = nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['TotalNitrogen 2007'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['cropland 2007'].sum()
KSNR2012 = nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['TotalNitrogen 2012'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['cropland 2012'].sum()
KSNR2017 = nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['TotalNitrogen 2017'].sum() / nitrogen_df2[nitrogen_df2['STATE'] == 'KANSAS']['cropland 2017'].sum()

print('KSNR1997:',KSNR1997,  'KSNR2002:',KSNR2002, 'KSNR2007:', KSNR2007, 'KSNR2012:', KSNR2012, 'KSNR2017:', KSNR2017)
print(f"\nKSNRavg: {(KSNR1997 + KSNR2002 + KSNR2007 + KSNR2012 + KSNR2017)/5}")


# # Map of Nitrogen Rate in kg used by acre of cropland

# In[11]:


get_ipython().run_line_magic('pip', 'install -U plotly')


# In[12]:


# Load the county boundary coordinates
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[ ]:


# Build the choropleth
import plotly.express as px
fig = px.choropleth(nitrogen_df2, 
    geojson=counties, 
    locations ='FIPS', 
    color='NitrogenRate 2007',
    color_continuous_scale= px.colors.sequential.Viridis[::-1],
    range_color=(40, 190),
    scope="usa"
    
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


fig.show()


# In[ ]:


# Build the choropleth
import plotly.express as px
fig = px.choropleth(nitrogen_df2, 
    geojson=counties, 
    locations ='FIPS', 
    color='NitrogenRate 2017',
    color_continuous_scale= px.colors.sequential.Viridis[::-1],
    range_color=(40, 190),
    scope="usa"
    
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


fig.show()

