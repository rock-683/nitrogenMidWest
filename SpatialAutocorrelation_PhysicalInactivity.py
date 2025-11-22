#!/usr/bin/env python
# coding: utf-8

# # Spatial Autocorrelation: Physical Inactivity
# 
# Physical Inactivity = Percentage of adults age 20 and over reporting no leisure-time physical activity.

# In[ ]:


pip install pysal==2.0.0


# In[ ]:


pip install esda


# In[ ]:


pip install splot   #this might take some time the first time


# In[ ]:


get_ipython().run_line_magic('pip', 'install -U plotly')


# In[ ]:


# Graphics
import matplotlib.pyplot as plt
import seaborn as sbn
from pysal.viz import splot
#from splot.esda import plot_moran

# Analysis
import geopandas as gpd  # Spatial data manipulation
import pandas as pd
import numpy as np
from pysal.explore import esda
from pysal.lib import weights
from numpy.random import seed


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
import plotly.figure_factory as ff
#https://plotly.com/python/builtin-colorscales/

from matplotlib import colors


# ### County Health Ranking data and Nitrogen data

# In[ ]:


county_data = pd.read_csv('NCH2017.csv')  # county_nitrogen + county_health


# ### Loading in geometry/polygon data for each county

# In[ ]:


geodata_county = gpd.read_file('MWCountyGeo.shp')


# In[ ]:


# Change geodata_county FIPS column type to match county_data FIPS column   
geodata_county ['FIPS'] = geodata_county['FIPS'].astype('int')

#Merge geometry/polygon to county_data
df = geodata_county.merge(county_data,on=['FIPS'],how='left')

#make sure it has correct projections (CRS)
df = gpd.GeoDataFrame(df, crs="EPSG:5071")

df.head(2)


# In[ ]:


# predictor variable
dfmw = df.dropna(subset=['Physical inactivity 2017 raw value'])
dfmw.head(2)


# In[ ]:


# Index table on variable of interest
db = dfmw.set_index("Physical inactivity 2017 raw value", drop=False)
db.crs


# In[ ]:


# Reprojecting from Decimal Degree to a planimetric projection (Spherical Mercator) 
# for enabling distance measures needed for spaital weights.
db2 = db.to_crs(epsg=3857)


# Here I am choosing contiguity/adjacency spatial weight, specifically **Queen contiguity**. I think because the data are counties, that adjacency/contguity spatial weights works well for this polygon data. Although, it may not present much of a difference here, I am choosing Queen over Rook as to include as neighboors those polygons that only share one or more vertices.

# In[ ]:


# spaital weights
w = weights.contiguity.Queen.from_dataframe(db2)

#standardize matrix
w.transform = 'R'

#finding the log transform of the data to get the "y"
y = db2['Physical inactivity 2017 raw value']


# In[ ]:


# Finding the spatial lag and adding column ‘lag’
db2['lag'] = weights.lag_spatial(w,y)


# Calculating a **global measure** of spatial autocorrelation - **Moran's I** 

# In[ ]:


#Moran's I and p-value
mi = esda.Moran(db2['lag'], w)
mi


# In[ ]:


mi.I


# In[ ]:


mi.p_sim


# `Physical inactivity 2017 raw value` has a moderate/modest positive spatial autocorrelation (.69). This means that there is a clustered pattern as opposed to a random or dispersed pattern. The p-value is .001, so we reject the null - the pattern reached significance. 

# In[ ]:


# Let's take a look at the Moran's I plot. Remember to look at the 4 quarters.
# finishing changing the code to match current variables

db2['lag'] = weights.spatial_lag.lag_spatial(w, db2['Physical inactivity 2017 raw value'])


# In[ ]:


# moran_scatterplot(mi);
f, ax = plt.subplots(1, figsize=(6, 6))
sbn.regplot(
    x="Physical inactivity 2017 raw value",
    y="lag",
    ci=None,
    data=db2,
    line_kws={"color": "r"},
)
ax.axvline(0, c="k", alpha=0.5)
ax.axhline(0, c="k", alpha=0.5)
ax.set_title("Moran Plot - % Leave")
plt.show()


# Looking more locally for spatial autocorrelation using the Local Indicator of Spatial Autocorrelation (**LISA**).

# In[ ]:


lisa = esda.Moran_Local(db2['lag'], w)


# Adding columns to our data for both significance and what quadrant the counties fell in.

# In[ ]:


# Break counties into significant or not
db2['significant'] = lisa.p_sim < 0.05
db2['significant'] = db2['significant'].astype('int').astype("str")


# Tag what quadrant they belong to
db2['quad'] = lisa.q


# In[ ]:


#reset index
db2.reset_index(drop=True, inplace=True)


# In[ ]:


# Pick only significant counties - assign '0' to non-significant

spots = []

for x in range(len(db2.significant)):
    for obs in db2['significant'][x]:
        if obs == "0":
            spots.append(0)
        if obs == '1':
            spots.append(db2['quad'][x])

db2['spots'] = spots

# check the unique values
db2.spots.unique()


# In[ ]:


#Mapping from value to name (as a dict)
spots_labels = {
    0: "Non-Significant",
    1: "HH",
    2: "LH",
    3: "LL",
    4: "HL",
}


# In[ ]:


# Create column in `db` with labels for each polygon
db2["labels"] = pd.Series(
    # First initialise a Series using values and `db` index
    spots,
    index=db2.index
    # Then map each value to corresponding label based
    # on the `spots_labels` mapping
).map(spots_labels)
# Print top for inspection
db2["labels"].value_counts()


# In[ ]:


#Getting Total Midwest Map

MidwestCountyMap = geodata_county[['FIPS','STATE_NAME','geometry']]

MidwestCountyMap = MidwestCountyMap.to_crs({'init':'epsg:3857'})


# In[ ]:


from matplotlib.colors import ListedColormap, BoundaryNorm

# Set up figure and axes
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 9))
# Make the axes accessible with single indexing


# Subplot 1 #
# Choropleth of local statistics


# Plot gray county base
base = MidwestCountyMap.plot(color='#EDECED', 
                      edgecolor='white',
ax=ax)

#Plot choropleth of local statistics
db2.plot(column='Physical inactivity 2017 raw value', 
         cmap='Greens', 
         edgecolor='white',
         linewidth=0.2,
         alpha=0.8,
         legend = True,
ax=base

)

# Figure styling #
# Set title to each subplot
ax.set_axis_off()
ax.set_title('Local Statistics: Phyiscal Inactivity'
    )
# Tight layout to minimise in-betwee white space
f.tight_layout()

#Save figure
#plt.savefig("mw_pia")
    
# Display the figure
plt.show()


# In[ ]:


from matplotlib.colors import ListedColormap, BoundaryNorm

# Set up figure and axes
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 9))
# Make the axes accessible with single indexing


# Subplot 2 #
# Quadrant categories

# Plot gray county base
base = MidwestCountyMap.plot(color='#EDECED', 
                      edgecolor='white',
ax=ax)

# Plot Quandrant colors 
db2.plot(column='labels', 
         cmap=ListedColormap(['#D7191C','#FDAE61', '#ABD9E9','#2C7BB6','#D3D3D3']), 
         edgecolor='white', 
         alpha=0.8,
         legend = True, 
ax=base
)

# Figure styling #
# Set title to each subplot
ax.set_axis_off()
ax.set_title(
    'Moran Cluster Map: Physical Inactivity Significant Hot Spots and Cold Spots'
    )
ax.get_legend().remove()
# Tight layout to minimise in-betwee white space
f.tight_layout()

#Save figure
plt.savefig("mw_pia_hscs")
    
# Display the figure
plt.show()

