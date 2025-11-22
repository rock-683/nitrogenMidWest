#!/usr/bin/env python
# coding: utf-8

# # Data Source and Data Carpentry

# county_nitrogen, county_health , and geodata_county

# In[12]:


import geopandas as gpd 
import pandas as pd
import numpy as np


# ### <font color=darkblue>Geogrpahic Scope: EPA Region 7 (Midwest) Data</font>
# The geographic scope for all datasets is Environmental Protection Agency (EPA) Region 7 (**Iowa, Kansas, Missouri, and Nebraska**) - Midwest.
# <br> https://www.epa.gov/aboutepa/epa-region-7-midwest

# # <font color=darkblue> Making county_nitrogen dataset</font>

# ### Reading in 1997 - 2017 Nitrogen Fertilizer and Nitrogen Manure Data 

# In[13]:


nitrogen_fertilizer = pd.read_csv('nitrogen_fertilizer.csv')
nitrogen_manure = pd.read_csv('nitrogen_manure.csv')


# ### Data Source
# 
# The nitrogen fertilizer and nitrogen manure datasets used in county_nitrogen table are from Falcone (2021). Citation: Falcone, J.A., 2021, Estimates of county-level nitrogen and phosphorus from fertilizer and manure from 1950 through 2017 in the conterminous United States: U.S. Geological Survey Open-File Report 2020–1153, 20 p., https://doi.org/10.3133/ofr20201153.
# 
# The Faclone (2021) data provides tabular county-level estimates of **kilograms** of nitrogen generated from two sources: (a) **fertilizer from commercial sources** and (b) **livestock-based manure**, for the period 1950 through 2017 for the conterminous United States - filtered for the midwest and years 1997 through 2017 for this project. Datasets collected during this time span are for intervals of approximately 5 years that coincide with the U.S. Department of Agriculture’s census years

# In[14]:


nitrogen_fertilizer.head(2)


# In[15]:


nitrogen_manure.head(2)


# In[16]:


# renaming fertilizer columns
nitrogen_fertilizer.columns = ['FIPS','fips-int','CountyName','StateAbbreviation','1997','2002','2007',
                  '2012','2017']

# melting dataset for tidy data
nitrogen_fertilizer_melt = pd.melt(nitrogen_fertilizer, id_vars = ["FIPS","fips-int","CountyName","StateAbbreviation"], 
                  var_name = 'Year', value_name = 'fertilizerNkg')


# renaming manure columns
nitrogen_manure.columns = ['FIPS','fips-int','CountyName','StateAbbreviation','1997','2002','2007',
                  '2012','2017']


# melting dataset for tidy data
nitrogen_manure_melt = pd.melt(nitrogen_manure, id_vars = ["FIPS","fips-int","CountyName","StateAbbreviation"], 
                               var_name = 'Year', value_name = 'manureNkg')


# In[17]:


nitrogen_fertilizer_melt.head(2)


# In[18]:


nitrogen_manure_melt.head(2)


# ### Reading in Cropland Treated Acres 

# In[20]:


#Acres of county Cropland Treated (Excluding Cropland Pastured)
cropland = pd.read_csv('cropland.csv')


# ### Data Source
# 
# Source: Acres of county cropland treated (excluding cropland pastured) used in the county_nitrogen table is derived from the the United States Department of Agriculture (USDA), National Agricultural Statistics Service (NASS). See https://www.nass.usda.gov/index.asp also accessible from the NASS Quick Stats online database at https://www.nass.usda.gov/Quick_Stats/ ; https://quickstats.nass.usda.gov/
# 
# This data was filtered for the midwest and years 1997 through 2017 for this project, and will be used to divide the sum of the nutrient mass by the cropland county area in the nitrogen_county table. 

# In[21]:


cropland.head(2)


# In[22]:


# Change cropland variable (str) to an interger (int)
cropland = cropland.dropna(subset=['cropland']) # Dropping missing values to change string to int
cropland = cropland[~cropland['cropland'].isin([' (D)'])]  #filtering out specific type of missing value (D) to change string to int
cropland['cropland'] = cropland['cropland']. str. replace(',','')  #taking out comma in thousands to change string to int
cropland['cropland'] = cropland['cropland'].astype("int") # Change acres value variable to an interger

# Combine State and County ANSI codes to make a complete ANSI code which will later be used to merge with FIPS
cropland['ANSI'] = cropland['State ANSI'].astype(str) + cropland['County ANSI'].astype(str).str.zfill(5)  #create leading zeros (0)
cropland = cropland[~cropland['ANSI'].isin(['200nan'])] #filtering out missing values
cropland['ANSI'] = cropland['ANSI'].astype(float).astype(int)  #converting to interger

# changing ANSI to FIPS (see detailed breakdown of each geographic idenitfier below)

cropland = cropland[['ANSI','State','Year','cropland']]
cropland.columns = ['FIPS', 'State', 'Year', 'cropland']
cropland.head()

cropland.head()


# #### FIPS and ANSI

# FIPS codes are typically used to identify geographic areas, such as states, counties, and metropolitan areas, for administrative and statistical purposes. These codes are assigned by the federal government and are standardized across all government agencies.
# 
# ANSI codes, on the other hand, are typically used to identify products, organizations, and other types of information. These codes are developed by the American National Standards Institute, which is a private, non-profit organization that develops voluntary standards for a variety of industries.
# 
# Because FIPS and ANSI codes are used for different purposes, there may be cases where a county or other geographic area has different codes in each system. For example, a county may have one FIPS code for administrative purposes and a different ANSI code for identifying a particular product or organization that is based in that county.
# 
# However, it's important to note that such cases of discrepancy between FIPS and ANSI codes are likely to be relatively rare, and most counties are likely to have consistent codes across both systems.

# In[23]:


# Merging nitrogen 1997,2002,2007, 2012, 2017 files and cropland_acres data 
Nfm = nitrogen_fertilizer_melt.merge(nitrogen_manure_melt, on=['FIPS','fips-int','CountyName','StateAbbreviation','Year'],how='left')

# Change year into string
cropland['Year'] = cropland['Year'].astype(str)  #converting to string

# Merging nitrogen 1997,2002,2007, 2012, 2017 files and cropland_acres data 
Nfmcl = Nfm.merge(cropland, on=['FIPS','Year'],how='left')
Nfmcl = Nfmcl[['FIPS','fips-int','State','StateAbbreviation','CountyName','Year','fertilizerNkg','manureNkg','cropland']]

#TotalNitrogen = Kilograms of Nitrogen (Fertilizer and Manure)
Nfmcl['TotalNitrogen'] = (Nfmcl['fertilizerNkg'] + Nfmcl['manureNkg'])
Nfmcl['NitrogenRate'] = (Nfmcl['fertilizerNkg'] + Nfmcl['manureNkg'])/Nfmcl['cropland']

Nfmcl.head()


# In[24]:


# delete probably 

MWNRAvg = Nfmcl[Nfmcl['State'] == 'MISSOURI']['TotalNitrogen'].sum() / Nfmcl[Nfmcl['State'] == 'MISSOURI']['cropland'].sum()

MWNRAvg


# In[ ]:


#Create a new data frame in the wide format that will pivot the nitrogen values into new columns
county_nitrogen = Nfmcl.pivot_table(index=['FIPS','State','CountyName'], columns='Year', values = ['TotalNitrogen','cropland','NitrogenRate'])

# join MultiIndex into one index 
county_nitrogen.columns = [' '.join(col).strip() for col in county_nitrogen.columns.values]

# reset the data frame index
county_nitrogen = county_nitrogen.reset_index()

county_nitrogen.head()


# In[ ]:


# check wich one you use in the trial -- or just don't include. 

county_nitrogen['AvgRate'] = (county_nitrogen['NitrogenRate 2007'] + county_nitrogen['NitrogenRate 2012'] + county_nitrogen['NitrogenRate 2017'])/3
county_nitrogen['CumulativeRate'] = (county_nitrogen['TotalNitrogen 2002'] + county_nitrogen['TotalNitrogen 2007'] + county_nitrogen['TotalNitrogen 2012'] + county_nitrogen['TotalNitrogen 2017']) /(county_nitrogen['cropland 2002'] + county_nitrogen['cropland 2007'] + county_nitrogen['cropland 2012'] + county_nitrogen['cropland 2017'])


# ## Final county_nitrogen dataset

# In[ ]:


county_nitrogen.head()


# # <font color=darkblue> county_health dataset</font>

# ## <font color=darkblue> County Health Ranking Data (Source Year: 2017) - Midwest/ EPA Region 7 </font>
# 
# County Health data where source year is 2017. This is a combination of county health ranking data from 2018 through 2023 release years. The release year is not necessarily comparable to the source year of the data. Therefore, to ensure that each feature aligned as close to the target year (2017) as possible, 6 county health ranking release years were used to generate one dataset where most metrics fit the desired timespan. Look at the data dictionary above to find county health ranking release year and source / source year of the data used. 
# 
# The 2017 dataset includes (a) **health outcomes** (premature death, poor or fair health, poor physical health days, poor mental health days, low birthweight), (b) **health behaviors** (smoking, obesity, food environment, physical inactivity, access to exercise, drinking, stis, teen births), (c) **Clinical Care** (uninsured, primary care physcians, mental health providers, preventable hospital stays, diabetes monitoring, mammography screening), (d) **Social and Economic Environment** (high school graduation, some college, unemployment, children in poverty, income inequality, children in single-parent households, social associatoins), and (e) **physical environment** (air pollution -particulate matter, drinking water violations, severe housing problems, driving alone to work, long commute - driving alone)
# 
# 
# ### Data Source
# 
# The data source is countyhealthrankings.org.
# https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation and  https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation/national-data-documentation-2010-2019

# ### 2017 Data Dictionary (dd)

# In[ ]:


chr_datadictionary2017 = pd.read_csv('chr_dd2017.csv')

pd.set_option("max_rows", None)
pd.set_option('display.max_colwidth', 255)

chr_datadictionary2017['Release Year'] = chr_datadictionary2017['Release Year'].astype(str).apply(lambda x: x.replace('.0',''))

chr_datadictionary2017


#new variables for 2023 County Health Ranking have been added including 
#Child Mortality, 
#Infant Mortality and 
#Low Birth Weight (% Low Birthweight 20142020)


# ### Reading in 2017 County Health Rankings Data - Midwest

# In[ ]:


pd.set_option('max_columns', None)
county_health = pd.read_csv('countyhealthranking2017.csv')
county_health.head()


# ## Merging Nitrogen and County Health Ranking data

# In[ ]:


county_nitrogen = county_nitrogen.rename(columns={'State': 'STATE'})

NCH2017 = county_nitrogen.merge(county_health,on=['FIPS'],how='left') 


# In[ ]:


NCH2017.head()


# In[ ]:


NCH2017.to_csv('NCH2017.csv',index=False)


# ### Geometry/Polygon Data for each County

# In[ ]:


geodata_county = gpd.read_file('MWCountyGeo.shp') 
geodata_county.head()


# In[ ]:


#Merge geometry/polygon to rest of midwest data
fipsmerge = geodata_county.merge(NCH2017,on=['FIPS'],how='left')

#make sure it has correct projections (CRS)
fipsmerge = gpd.GeoDataFrame(fipsmerge, crs="EPSG:5071")

fipsmerge.head(1)

