#!/usr/bin/env python
# coding: utf-8

# # Geographically Weighted Regression (GWR) / Multiscale Geographically Weighted Regression (MGWR): 
# Dependent Variable = Low Birthweight

# In[ ]:


pip install pysal==2.0.0


# In[ ]:


pip install mgwr


# In[ ]:


import geopandas as gpd
import seaborn as sbn
import pandas as pd
import numpy as np
import pysal
import libpysal as ps
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.utils import compare_surfaces, truncate_colormap
from mgwr.utils import shift_colormap
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Polygon
from pysal.lib import cg as geometry
from pysal.lib import weights
import statistics


# ## Loading, Merging and Prepping Datasets

# ### County Health Ranking data and Nitrogen data

# In[ ]:


county_data = pd.read_csv('NCH2017.csv')  # county_nitrogen + county_health


# In[ ]:


county_data.head()


# ### Geometry/Polygon Data 
# for each county

# In[ ]:


geodata_county = gpd.read_file('MWCountyGeo.shp')
geodata_county.head()


# In[ ]:


# Change geodata_county FIPS column type to match county_data FIPS column   
geodata_county ['FIPS'] = geodata_county['FIPS'].astype('int')

#Merge geometry/polygon to county_data
df = geodata_county.merge(county_data,on=['FIPS'],how='left')

#make sure it has correct projections (CRS)
df = gpd.GeoDataFrame(df, crs="EPSG:5071")

df.head(2)


# ### Filter 
# 
# to target area and selected variables

# In[ ]:


#selected variables
dfmw = df[['FIPS','NAME','STATE_NAME','StateAbbreviation',
      'NitrogenRate 2017','percent Rural 2010 raw value', 'Preventable hospital stays 2017 raw value',
            'Severe housing cost burden 20132017 raw value','Food environment 20152017 index raw value',
            'Median household income 2017 raw value','Physical inactivity 2017 raw value',
            'percent Non-Hispanic African American2017  raw value','Food insecurity 2017 raw value',
            'Mammography screening 2017 raw value','Uninsured adults 2017 raw value',
            'percent below 18 years of age 2017 raw value','Uninsured 2017 raw value',
            '% Low Birthweight 20142020','geometry']] 


# In[ ]:


#checking for missing values
dfmw.isna().sum()


# In[ ]:


#county count
len(dfmw)


# In[ ]:


# drop rows with missing values
dfmw = dfmw.dropna()

#county count
len(dfmw)


# In[ ]:


# generate plot of Midwest counties
fig, ax = plt.subplots(figsize = (12, 12))
dfmw.plot(ax=ax, **{"edgecolor": "black", "facecolor": "white"})
dfmw.centroid.plot(ax = ax, c = "black")
plt.show()


# ### Adding Centroids

# In[ ]:


# Adding centroids to county locations so that inter-county distances can be computed within the GWR routine
dfmw["c"] = dfmw.centroid.astype('str')

# Splitting 'c' column
dfmw[['point', 'Point_X','Point_Y']] = dfmw['c'].str.split(' ', 2, expand=True)


dfmw['Point_X'] = dfmw['Point_X'].str.replace(r'[()]',"").astype('float').round()
dfmw['Point_Y'] = dfmw['Point_Y'].str.replace(r'[()]',"").astype('float').round()

#drop a couple of columns
dfmw = dfmw.drop(columns = ['c','point'])


# In[ ]:


dfmw.head()


# ### Preparing LBW dataset inputs

# In[ ]:


#Prepare LBW dataset inputs

mw_y = dfmw['% Low Birthweight 20142020'].values.reshape((-1, 1))
mw_X = dfmw[['NitrogenRate 2017','percent Rural 2010 raw value', 'Preventable hospital stays 2017 raw value',
            'Physical inactivity 2017 raw value', 'percent Non-Hispanic African American2017  raw value', 
                'Food insecurity 2017 raw value']].values
u = dfmw["Point_X"]
v = dfmw["Point_Y"]
mw_coords = list(zip(u, v))


# In[ ]:


chr_dd2017 = pd.read_csv('/dsa/groups/capstonesp2023/online/Team07/chr2017_US_datadictionary.csv')

pd.set_option("max_rows", None)
pd.set_option('display.max_colwidth', 255)

chr_dd2017['Release Year'] = chr_dd2017['Release Year'].astype(str).apply(lambda x: x.replace('.0',''))

#chr_dd2017
chr_dd2017.iloc[[7,83,12,22,75,50],:]


# ### Standardizing the Variables
# 
# In order to compare each of the bandwidths obtained from an MGWR model, it is necessary to standardize the dependent and independent variables so that they are centered at zero and based on the same range of variation. Otherwise it may be difficult to objectively compare the estimated bandwidths because it is possible that they are also representative of the scale and variation of the independent variables.

# In[ ]:


#Standardize~variables Midwest dataset

mw_X = (mw_X - mw_X.mean(axis = 0)) / mw_X.std(axis = 0)
mw_y = (mw_y - mw_y.mean(axis = 0)) / mw_y.std(axis = 0)


# ## Geographically Weighted Regression (GWR): 
# Dependent Variable = LBW

# In[ ]:


#Calibrate GWR model

gwr_selector = Sel_BW(mw_coords, mw_y, mw_X)
gwr_bw = gwr_selector.search(bw_min=2)
print(gwr_bw)

gwr_results = GWR(mw_coords, mw_y, mw_X, gwr_bw, kernel='bisquare').fit()


# In[ ]:


gwr_results.summary()


# params 
# <br>[:,0] = Intercept
# <br>[:,1] = Nitrogen Rate, 
# <br>[:,2] = Percent Rural, 
# <br>[:,3] = Preventable Hospital, 
# <br>[;,4] = Physical Inactivity 
# <br>[;,5] = Percent African American, 
# <br>[;,6] = Food Insecurity

# In[ ]:


#Local model fit
dfmw["R2"] = gwr_results.localR2
dfmw.plot("R2", 
                   legend = True, 
                   cmap='viridis_r')
ax = plt.gca() 
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig("local_R2")
plt.show()


# ## Mutliscale Geographically Weighted Regression (MGWR): 
# Dependent Variable = Low Birthweight

# In a multiscale geographical weighted regression (MGWR) model, the bandwidth refers to the distance threshold or smoothing parameter used to determine the geographic extent of the local relationships between the dependent variable and the independent variables at each location in the study area.
# 
# In other words, the bandwidth specifies the size of the local neighborhood around each location, within which the relationship between the response variable and the predictors is modeled. **The bandwidth determines the level of spatial autocorrelation that is taken into account in the model, with smaller bandwidths capturing more local spatial relationships and larger bandwidths capturing more global spatial relationships**.
# 
# Choosing an appropriate bandwidth is critical in MGWR, as it can affect the quality of the estimated coefficients, the accuracy of the predictions, and the interpretation of the spatial relationships between the variables. A common approach to selecting the bandwidth is to use cross-validation techniques to identify the value that maximizes the model performance or goodness-of-fit criteria.

# In[ ]:


#Unique manual bandwidths, gaussian
mgwr_selector = Sel_BW(mw_coords, mw_y, mw_X, multi = True)
mgwr_bw = mgwr_selector.search(multi_bw_min = [15, 50, 75, 100, 50, 370, 360],
                         multi_bw_max = [15, 50, 75, 100, 50, 370, 360])
print(mgwr_bw)

mgwr_results = MGWR(mw_coords, mw_y, mw_X, mgwr_selector, kernel='gaussian').fit()


# params 
# <br>[:,0] = Intercept
# <br>[:,1] = Nitrogen Rate, 
# <br>[:,2] = Percent Rural, 
# <br>[:,3] = Preventable Hospital, 
# <br>[;,4] = Physical Inactivity 
# <br>[;,5] = Percent African American, 
# <br>[;,6] = Food Insecurity

# In[ ]:


# NitrogentRate 2017 = .654
# NitrogenRate 2012 = .647
# NitrogenRate 2007 = .652
# NitrogenRate 2002 = .648
# 2012 + 2017 cummulative nitrogen rate = .651
# 2007 + 2012 + 2017 cummulative nitrogen rate = .651

## could test TotalNitrogen in the same way, don't have to use the cropland but you don't have to, that one doesn't make as much sense to me
# doesn't have as much reasoning behind it

mgwr_results.summary()


# In[ ]:


import matplotlib.pyplot as plt

# feature names and t-values
features = ['% African American', 'Food Insecurity', 'Preventable Hospital Stays', 'Nitrogen Rate', '% Rural', 'Physical Inactivity']
t_values = [2.001, 2.015, 2.317, 2.386, 2.39, 2.505]

# create figure and axis objects
fig, ax = plt.subplots(figsize=(8, 5))

# plot horizontal bars
bars = ax.barh(features, t_values, color=['#b0b7bf' if x != 2.386 else '#44546A' for x in t_values])

    
# Add text labels to bars
#for i, v in enumerate(t_values):
 #   ax.text(v + .1, i, str(v), color='black', fontsize=12)
    
# Add text labels inside bars
for bar in bars:
    xval = bar.get_width()
    ax.text(xval*.85, bar.get_y() + bar.get_height()/2., str(xval), va='center', color='white', fontsize=12)

# set axis labels and title
ax.set_title('T-Value',fontsize=16)

# remove spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(top=False, right=False)
ax.tick_params(left=False, bottom=False)

# Remove x-ticks and labels
ax.set_xticks([])
ax.yaxis.set_tick_params(labelsize=17)

# adjust layout
plt.tight_layout()

#Save figure
plt.savefig("featureimportance1")

# show plot
plt.show()


# In[ ]:


dfmw['mgwr_intercept'] = mgwr_results.params[:,0] #X0 - intercept
dfmw['mgwr_nitrogen'] = mgwr_results.params[:,1] #X1 - Nitrogen parameter estimates
dfmw['mgwr_rural'] = mgwr_results.params[:,2] #X2 - Percent Rural 2010 parameter estimates
dfmw['mgwr_phs'] = mgwr_results.params[:,3] #X3 - Preventable hospital stays 2017 
dfmw['mgwr_pia'] = mgwr_results.params[:,4] #X4 - Physical inactivity 2017
dfmw['mgwr_aa'] = mgwr_results.params[:,5] #X5 - percent Non-Hispanic African American 2017
dfmw['mgwr_foodis'] = mgwr_results.params[:,6] #X6 - Food insecurity 2017


# In[ ]:


n = statistics.mean(abs(dfmw['mgwr_nitrogen']))
r = statistics.mean(abs(dfmw['mgwr_rural']))
phs = statistics.mean(abs(dfmw['mgwr_phs']))
pia = statistics.mean(abs(dfmw['mgwr_pia']))
aa = statistics.mean(abs(dfmw['mgwr_aa']))
f = statistics.mean(abs(dfmw['mgwr_foodis']))

print('Nitrogen:',n, '\nPhysical Inactivity:',pia, '\nrural:',r, '\nFood Insecurity:',f,
      '\npreventable hospital stays:',phs,
      '\nPercent African American:',aa, )


# In[ ]:


## Feature Importance, average magnitude from parameter estimates

# feature names and Mean Magnitude of Parameter Estimates
features = ['% African American', 'Preventable Hospital Stays','Food Insecurity','% Rural','Physical Inactivity','Nitrogen Rate']
magnitude = [.1, .13, .15, .16, .21, 0.26]

# create figure and axis objects
fig, ax = plt.subplots(figsize=(8, 5))

# plot horizontal bars
bars = ax.barh(features, magnitude, color=['#b0b7bf' if x != 0.26 else '#44546A' for x in magnitude])

    
# Add text labels to bars
#for i, v in enumerate(magnitude):
 #   ax.text(v + .01, i, str(v), color='black', fontsize=30)
    
    
# Add text labels inside bars
for bar in bars:
    xval = bar.get_width()
    ax.text(xval*.75, bar.get_y() + bar.get_height()/2., str(xval), va='center', color='white', fontsize=12)


# set axis labels and title
ax.set_title('Mean Magnitude of Parameter Estimates', fontsize=16)

# remove spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(top=False, right=False)
ax.tick_params(left=False, bottom=False)

# Remove x-ticks and labels
ax.set_xticks([])
ax.yaxis.set_tick_params(labelsize=17)

# adjust layout
plt.tight_layout()

#Save figure
plt.savefig("featureimportance2")

# show plot
plt.show()


# params 
# <br>[:,0] = Intercept
# <br>[:,1] = Nitrogen Rate, 
# <br>[:,2] = Percent Rural, 
# <br>[:,3] = Preventable Hospital, 
# <br>[;,4] = Physical Inactivity 
# <br>[;,5] = Percent African American, 
# <br>[;,6] = Food Insecurity

# ## Testing Assumptions
# 
# ### Local Multicollinearity
# 
# Though there are many tools available to evaluate multicollinearity amongst explanatory variables for traditional regression models, some extra care is needed for local models that borrow data from nearby locations. Within each local model, there may be higher levels of collinearity than is present in the dataset as a whole. Higher levels of collinearity are associated with problems such as estimate instability, unintuitive parameter signs, high R2
#  diagnostics despite few or no significant parameters, and inflated standard errors for parameter estimates. As a result, diagnostic tools have been designed to detect levels of local multicollinearity, including local Correlation Coefficients (CC), local Variation Inflation Factors (VIF), local Condition Number (CN), and local Variation Decomposition Proportions (VDP).
#  
# Here we are using VIF. Each local measure has a rule of thumb that indicates that there might be an issue due to multicollinearity: VIF higher than 10 indicate multicollinearity in some measure. However, these rules are not absolute and obtaining lower values does not mean collinearity is innocuous, nor does obtaining larger values guarantee collinearity is indeed problematic. In addition, local VIF’s do not consider the local intercept term. The maps generated below demonstrate local VIF’s.
# 
# **The VIF's are all well below 10 indicating that that collinearity is not problemicatic for any of the calibration locations.** 
# 
# In addition, it has been demonstrated that multicollinearity is not inherently more problematic in GWR than a traditional regression and some of the patterns theorized to be associated with multicollinearity may be indicative of reality or due to scale misspecification.
# 
# VIF for multicollinearity
# <br>[:,0] = Nitrogen Rate, 
# <br>[:,1] = Percent Rural, 
# <br>[:,2] = Preventable Hospital, 
# <br>[;,3] = Physical Inactivity 
# <br>[;,4] = Percent African American, 
# <br>[;,5] = Food Insecurity

# In[ ]:



# Local Multicollinearity
LCC, VIF, CN, VDP = gwr_results.local_collinearity()


dfmw["vif_nitrogen"] = VIF[:, 0]
dfmw["vif_rural"] = VIF[:, 1]
dfmw["vif_pha"] = VIF[:, 2]
dfmw["vif_pia"] = VIF[:, 3]
dfmw["vif_aa"] = VIF[:, 4]
dfmw["vif_foodis"] = VIF[:, 5]

fig, ax = plt.subplots(2, 3, figsize = (16, 8))
dfmw["vif"] = VIF[:, 0]
dfmw.plot("vif", ax = ax[0,0], legend = True)
ax[0,0].set_title("VIF: " + 'Nitrogen')
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].get_yaxis().set_visible(False)

dfmw["vif"] = VIF[:, 1]
dfmw.plot("vif", ax = ax[0,1], legend = True)
ax[0,1].set_title("VIF: " + 'Rural')
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)

dfmw["vif"] = VIF[:, 2]
dfmw.plot("vif", ax = ax[0,2], legend = True)
ax[0,2].set_title("VIF: " + 'Preventable Hospital Stays')
ax[0,2].get_xaxis().set_visible(False)
ax[0,2].get_yaxis().set_visible(False)


dfmw["vif"] = VIF[:, 3]
dfmw.plot("vif", ax = ax[1,0], legend = True)
ax[1,0].set_title("VIF: " + 'Physical Inactivity')
ax[1,0].get_xaxis().set_visible(False)
ax[1,0].get_yaxis().set_visible(False)

dfmw["vif"] = VIF[:, 4]
dfmw.plot("vif", ax = ax[1,1], legend = True)
ax[1,1].set_title("VIF: " + 'Percent African American')
ax[1,1].get_xaxis().set_visible(False)
ax[1,1].get_yaxis().set_visible(False)


dfmw["vif"] = VIF[:, 5]
dfmw.plot("vif", ax = ax[1,2], legend = True)
ax[1,2].set_title("VIF: " + 'Food Insecurity')
ax[1,2].get_xaxis().set_visible(False)
ax[1,2].get_yaxis().set_visible(False)


# ### Inference on Individual Parameter Estimates
# 
# As with GWR, it is necessary to apply the modified hypothesis testing framework described above. However, in the case of MGWR, it is possible to extend the testing framework to formulate a covariate-specific corrected hypothesis test for each surface of parameter estimates. This novel methodology is described in [7] and the necessary functionality is not currently available in other software implementations other than mgwr. 
# 
# In MGWR, the hat matrix, S, that maps the observed dependent variable onto the fitted values of the dependent variable, can be decomposed into covariate-specific contributions, Rj. With this, it is possible to compute a distinct measure of the effective number of parameters (ENP) for each parameter surface.
# 
# The default behavior in mgwr is to use αj to compute a covariate-specific critical t-value for hypothesis testing. It is possible to inspect each ENPj, αj, and the adjusted t-values as follows:

# In[ ]:


#Covariate-specific ENP
print(mgwr_results.ENP_j)

#Covrariate-specific adjusted alpha at 95% CI
print(mgwr_results.adj_alpha_j[:, 1])

#Covariate-specific adjusted critical t-value
print(mgwr_results.critical_tval())


# ## Parameter Estimates for each Parameter
# 
# params 
# <br>[:,1] = Nitrogen Rate, 
# <br>[:,2] = Percent Rural, 
# <br>[:,3] = Preventable Hospital, 
# <br>[;,4] = Physical Inactivity 
# <br>[;,5] = Percent African American, 
# <br>[;,6] = Food Insecurity

# In[ ]:


#Prepare GWR results for mapping

#Add GWR parameters to GeoDataframe
dfmw['gwr_intercept'] = gwr_results.params[:,0] #X0 - intercept
dfmw['gwr_nitrogen'] = gwr_results.params[:,1] #X1 - Nitrogen parameter estimates
dfmw['gwr_rural'] = gwr_results.params[:,2] #X2 - Percent Rural 2010 parameter estimates
dfmw['gwr_phs'] = gwr_results.params[:,3] #X3 - Preventable hospital stays 2017 
dfmw['gwr_pia'] = gwr_results.params[:,4] #X4 - Physical inactivity 2017
dfmw['gwr_aa'] = gwr_results.params[:,5] #X5 - percent Non-Hispanic African American 2017
dfmw['gwr_foodis'] = gwr_results.params[:,6] #X6 - Food insecurity 2017



#Obtain t-vals filtered based on multiple testing correction
gwr_filtered_t = gwr_results.filter_tvals()


# In[ ]:


#Prepare MGWR results for mapping

#Add MGWR parameters to GeoDataframe
dfmw['mgwr_intercept'] = mgwr_results.params[:,0] #X0 - intercept
dfmw['mgwr_nitrogen'] = mgwr_results.params[:,1] #X1 - Nitrogen parameter estimates
dfmw['mgwr_rural'] = mgwr_results.params[:,2] #X2 - Percent Rural 2010 parameter estimates
dfmw['mgwr_phs'] = mgwr_results.params[:,3] #X3 - Preventable hospital stays 2017 
dfmw['mgwr_pia'] = mgwr_results.params[:,4] #X4 - Physical inactivity 2017
dfmw['mgwr_aa'] = mgwr_results.params[:,5] #X5 - percent Non-Hispanic African American 2017
dfmw['mgwr_foodis'] = mgwr_results.params[:,6] #X6 - Food insecurity 2017



#Obtain t-vals filtered based on multiple testing correction
mgwr_filtered_t = mgwr_results.filter_tvals()


# ## Intercept
# 
# Variable                   Mean        STD        Min     Median        Max
# -------------------- ---------- ---------- ---------- ---------- ----------
# X0                       -0.062      0.398     -0.892     -0.097      1.082

# In[ ]:


#Comparison maps of GWR vs. MGWR parameter surfaces where the grey units pertain to statistically insignificant parameters

#Prep plot and add axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
ax0 = axes[0]
ax0.set_title('GWR Intercept Surface (BW: ' + str(gwr_bw) +')', fontsize=40)
ax1 = axes[1]
ax1.set_title('MGWR Intercept Surface (BW: ' + str(mgwr_bw[0]) +')', fontsize=40)

#Set color map
cmap = plt.cm.inferno

#Find min and max values of the two combined datasets
gwr_min = dfmw['gwr_intercept'].min()
gwr_max = dfmw['gwr_intercept'].max()
mgwr_min = dfmw['mgwr_intercept'].min()
mgwr_max = dfmw['mgwr_intercept'].max()
vmin = np.min([gwr_min, mgwr_min])
vmax = np.max([gwr_max, mgwr_max])

#If all values are negative use the negative half of the colormap
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
#If all values are positive use the positive half of the colormap
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
#Otherwise, there are positive and negative values so the colormap so zero is the midpoint
else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

#Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

#Plot GWR parameters
dfmw.plot('gwr_intercept', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (gwr_filtered_t[:,0] == 0).any():
    dfmw[gwr_filtered_t[:,0] == 0].plot(color='lightgrey', ax=ax0, **{'edgecolor':'black'})

#Plot MGWR parameters
dfmw.plot('mgwr_intercept', cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (mgwr_filtered_t[:,0] == 0).any():
    dfmw[mgwr_filtered_t[:,0] == 0].plot(color='lightgrey', ax=ax1, **{'edgecolor':'black'})
 
#Set figure options and plot 
fig.tight_layout()    
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=50) 
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
plt.show()


# ## Nitrogen Rate
# 
# Variable                   Mean        STD        Min     Median        Max
# -------------------- ---------- ---------- ---------- ---------- ----------
# X1                        0.016      0.335     -0.759      0.025      1.062

# In[ ]:


#Comparison maps of GWR vs. MGWR parameter surfaces where the grey units pertain to statistically insignificant parameters

#Prep plot and add axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
ax0 = axes[0]
ax0.set_title('GWR Nitrogen Rate Surface (BW: ' + str(gwr_bw) +')', fontsize=40)
ax1 = axes[1]
ax1.set_title('MGWR Nitrogen Rate Surface (BW: ' + str(mgwr_bw[1]) +')', fontsize=40)

#Set color map
cmap = plt.cm.inferno  
#cmap = plt.cm.get_cmap('inferno', 5)

#Find min and max values of the two combined datasets
gwr_min = dfmw['gwr_nitrogen'].min()
gwr_max = dfmw['gwr_nitrogen'].max()
mgwr_min = dfmw['mgwr_nitrogen'].min()
mgwr_max = dfmw['mgwr_nitrogen'].max()
vmin = np.min([gwr_min, mgwr_min])
vmax = np.max([gwr_max, mgwr_max])

#If all values are negative use the negative half of the colormap
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
#If all values are positive use the positive half of the colormap
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
#Otherwise, there are positive and negative values so the colormap so zero is the midpoint
else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

#Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

#Plot GWR parameters
dfmw.plot('gwr_nitrogen', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (gwr_filtered_t[:,1] == 0).any():
    dfmw[gwr_filtered_t[:,1] == 0].plot(color='lightgrey', ax=ax0, **{'edgecolor':'black'})

#Plot MGWR parameters
dfmw.plot('mgwr_nitrogen', cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (mgwr_filtered_t[:,1] == 0).any():
    dfmw[mgwr_filtered_t[:,1] == 0].plot(color='lightgrey', ax=ax1, **{'edgecolor':'black'})
 
#Set figure options and plot 
fig.tight_layout()    
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=50) 
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
plt.show()

### Nitrogen

More specifically, our results indicated considerable spatial heterogeneity in low birth weight rates along the low to high nitrogen rate gradient. In parts of Iowa and west Nebraska, as nitrogen rates increase, rates of low birth weight decrease. Contrastingly, in Kansas, central Nebraska and northern Missouri as nitrogen rates increase so does low birth weight.   

These findings indicate that the impact of nitrogen rate (amount of nitrogen from fertilizer divided by amount of cropland treated in each county) varies by county. This could be due to factors in each county such as how they deal with nitrogen run-off, types of water safety measures in place, crop rotation and other agricultural practices that could affect how much nitrogen polutes drinking water.
# ## Percent Rural
# 
# Variable                   Mean        STD        Min     Median        Max
# -------------------- ---------- ---------- ---------- ---------- ----------
# X2                       -0.146      0.113     -0.383     -0.156      0.171

# In[ ]:


#Comparison maps of GWR vs. MGWR parameter surfaces where the grey units pertain to statistically insignificant parameters

#Prep plot and add axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
ax0 = axes[0]
ax0.set_title('GWR Percent Rural Surface (BW: ' + str(gwr_bw) +')', fontsize=40)
ax1 = axes[1]
ax1.set_title('MGWR Percent Rural Surface (BW: ' + str(mgwr_bw[2]) +')', fontsize=40)

#Set color map
cmap = plt.cm.inferno

#Find min and max values of the two combined datasets
gwr_min = dfmw['gwr_rural'].min()
gwr_max = dfmw['gwr_rural'].max()
mgwr_min = dfmw['mgwr_rural'].min()
mgwr_max = dfmw['mgwr_rural'].max()
vmin = np.min([gwr_min, mgwr_min])
vmax = np.max([gwr_max, mgwr_max])

#If all values are negative use the negative half of the colormap
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
#If all values are positive use the positive half of the colormap
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
#Otherwise, there are positive and negative values so the colormap so zero is the midpoint
else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

#Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

#Plot GWR parameters
dfmw.plot('gwr_rural', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (gwr_filtered_t[:,2] == 0).any():
    dfmw[gwr_filtered_t[:,2] == 0].plot(color='lightgrey', ax=ax0, **{'edgecolor':'black'})

#Plot MGWR parameters
dfmw.plot('mgwr_rural', cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (mgwr_filtered_t[:,2] == 0).any():
    dfmw[mgwr_filtered_t[:,2] == 0].plot(color='lightgrey', ax=ax1, **{'edgecolor':'black'})
 
#Set figure options and plot 
fig.tight_layout()    
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=50) 
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
plt.show()

### Rural

In Iowa, Kansas, eastern Nebraska, and north west Missouri, as percent rural population increases, rates of low birth weight decrease. In south central Nebraska as percent rural population increases so does low birth weight.  
# ## Preventable Hospital Stays
# 
# Variable                   Mean        STD        Min     Median        Max
# -------------------- ---------- ---------- ---------- ---------- ----------
# X3                        0.122      0.079     -0.094      0.137      0.293

# In[ ]:


#Comparison maps of GWR vs. MGWR parameter surfaces where the grey units pertain to statistically insignificant parameters

#Prep plot and add axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
ax0 = axes[0]
ax0.set_title('GWR Preventable Hospital Stay Surface (BW: ' + str(gwr_bw) +')', fontsize=40)
ax1 = axes[1]
ax1.set_title('MGWR Preventable Hospital Stay Surface (BW: ' + str(mgwr_bw[3]) +')', fontsize=40)

#Set color map
cmap = plt.cm.inferno

#Find min and max values of the two combined datasets
gwr_min = dfmw['gwr_phs'].min()
gwr_max = dfmw['gwr_phs'].max()
mgwr_min = dfmw['mgwr_phs'].min()
mgwr_max = dfmw['mgwr_phs'].max()
vmin = np.min([gwr_min, mgwr_min])
vmax = np.max([gwr_max, mgwr_max])

#If all values are negative use the negative half of the colormap
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
#If all values are positive use the positive half of the colormap
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
#Otherwise, there are positive and negative values so the colormap so zero is the midpoint
else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

#Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

#Plot GWR parameters
dfmw.plot('gwr_phs', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (gwr_filtered_t[:,3] == 0).any():
    dfmw[gwr_filtered_t[:,3] == 0].plot(color='lightgrey', ax=ax0, **{'edgecolor':'black'})

#Plot MGWR parameters
dfmw.plot('mgwr_phs', cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (mgwr_filtered_t[:,3] == 0).any():
    dfmw[mgwr_filtered_t[:,3] == 0].plot(color='lightgrey', ax=ax1, **{'edgecolor':'black'})
 
#Set figure options and plot 
fig.tight_layout()    
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=50) 
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
plt.show()

### Preventable Hospital Stays

In Missouri, west Iowa, east Nebraska and east Kansas, as percentage of preventable hospital stays increases so does percent low birth weight.
# ## Physical Inactivity 
# Percentage of adults age 20 and over reporting no leisure-time physical activity.
# 
# Variable                   Mean        STD        Min     Median        Max
# -------------------- ---------- ---------- ---------- ---------- ----------
# X4                        0.186      0.180     -0.316      0.182      0.720

# In[ ]:


#Comparison maps of GWR vs. MGWR parameter surfaces where the grey units pertain to statistically insignificant parameters

#Prep plot and add axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
ax0 = axes[0]
ax0.set_title('GWR Physical Inactivity Surface (BW: ' + str(gwr_bw) +')', fontsize=40)
ax1 = axes[1]
ax1.set_title('MGWR Physical Inactivity Surface (BW: ' + str(mgwr_bw[4]) +')', fontsize=40)

#Set color map
cmap = plt.cm.inferno

#Find min and max values of the two combined datasets
gwr_min = dfmw['gwr_pia'].min()
gwr_max = dfmw['gwr_pia'].max()
mgwr_min = dfmw['mgwr_pia'].min()
mgwr_max = dfmw['mgwr_pia'].max()
vmin = np.min([gwr_min, mgwr_min])
vmax = np.max([gwr_max, mgwr_max])

#If all values are negative use the negative half of the colormap
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
#If all values are positive use the positive half of the colormap
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
#Otherwise, there are positive and negative values so the colormap so zero is the midpoint
else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

#Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

#Plot GWR parameters
dfmw.plot('gwr_pia', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (gwr_filtered_t[:,4] == 0).any():
    dfmw[gwr_filtered_t[:,4] == 0].plot(color='lightgrey', ax=ax0, **{'edgecolor':'black'})

#Plot MGWR parameters
dfmw.plot('mgwr_pia', cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (mgwr_filtered_t[:,4] == 0).any():
    dfmw[mgwr_filtered_t[:,4] == 0].plot(color='lightgrey', ax=ax1, **{'edgecolor':'black'})
 
#Set figure options and plot 
fig.tight_layout()    
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=50) 
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
plt.show()

### Physical Inactivity

In Iowa, Kansas, southern Missouri and east Nebraska as percentage of physical inactivity increases so does percent low birth weight. In central south Nebraska, as percentage of physical inactivity increases low birth weight decreases. Hot spots and cold spots for physical inactivity and low birth weight overlap in Iowa and Missouri.
# ## Percent African American
# 
# Variable                   Mean        STD        Min     Median        Max
# -------------------- ---------- ---------- ---------- ---------- ----------
# X5                        0.083      0.004      0.077      0.083      0.090

# In[ ]:


#Comparison maps of GWR vs. MGWR parameter surfaces where the grey units pertain to statistically insignificant parameters

#Prep plot and add axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
ax0 = axes[0]
ax0.set_title('GWR Percent African American Surface (BW: ' + str(gwr_bw) +')', fontsize=40)
ax1 = axes[1]
ax1.set_title('MGWR Percent African American Surface (BW: ' + str(mgwr_bw[5]) +')', fontsize=40)

#Set color map
cmap = plt.cm.autumn

#Find min and max values of the two combined datasets
gwr_min = dfmw['gwr_aa'].min()
gwr_max = dfmw['gwr_aa'].max()
mgwr_min = dfmw['mgwr_aa'].min()
mgwr_max = dfmw['mgwr_aa'].max()
vmin = np.min([gwr_min, mgwr_min])
vmax = np.max([gwr_max, mgwr_max])

#If all values are negative use the negative half of the colormap
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
#If all values are positive use the positive half of the colormap
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
#Otherwise, there are positive and negative values so the colormap so zero is the midpoint
else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

#Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

#Plot GWR parameters
dfmw.plot('gwr_aa', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (gwr_filtered_t[:,5] == 0).any():
    dfmw[gwr_filtered_t[:,5] == 0].plot(color='lightgrey', ax=ax0, **{'edgecolor':'black'})

#Plot MGWR parameters
dfmw.plot('mgwr_aa', cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (mgwr_filtered_t[:,5] == 0).any():
    dfmw[mgwr_filtered_t[:,5] == 0].plot(color='lightgrey', ax=ax1, **{'edgecolor':'black'})
 
#Set figure options and plot 
fig.tight_layout()    
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=50) 
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
plt.show()

### Percent African American

The findings demonstrated that the african american population was an influential factor in explaining spatial variation in low birth weight particulary, in Iowa, Nebraska and west Kansas as percent African American icreases, percent low birth weight also increases. 

# ## Food Insecurity
# Percentage of population who lack adequate access to food.
# 
# Variable                   Mean        STD        Min     Median        Max
# -------------------- ---------- ---------- ---------- ---------- ----------
# X6                        0.149      0.022      0.112      0.146      0.191

# In[ ]:


#Comparison maps of GWR vs. MGWR parameter surfaces where the grey units pertain to statistically insignificant parameters

#Prep plot and add axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
ax0 = axes[0]
ax0.set_title('GWR Food Insecurity Surface (BW: ' + str(gwr_bw) +')', fontsize=40)
ax1 = axes[1]
ax1.set_title('MGWR Food Insecurity Surface (BW: ' + str(mgwr_bw[6]) +')', fontsize=40)

#Set color map
cmap = plt.cm.inferno

#Find min and max values of the two combined datasets
gwr_min = dfmw['gwr_foodis'].min()
gwr_max = dfmw['gwr_foodis'].max()
mgwr_min = dfmw['mgwr_foodis'].min()
mgwr_max = dfmw['mgwr_foodis'].max()
vmin = np.min([gwr_min, mgwr_min])
vmax = np.max([gwr_max, mgwr_max])

#If all values are negative use the negative half of the colormap
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
#If all values are positive use the positive half of the colormap
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
#Otherwise, there are positive and negative values so the colormap so zero is the midpoint
else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

#Create scalar mappable for colorbar and stretch colormap across range of data values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

#Plot GWR parameters
dfmw.plot('gwr_foodis', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (gwr_filtered_t[:,6] == 0).any():
    dfmw[gwr_filtered_t[:,6] == 0].plot(color='lightgrey', ax=ax0, **{'edgecolor':'black'})

#Plot MGWR parameters
dfmw.plot('mgwr_foodis', cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **{'edgecolor':'black', 'alpha':.65})
#If there are insignificnt parameters plot gray polygons over them
if (mgwr_filtered_t[:,6] == 0).any():
    dfmw[mgwr_filtered_t[:,6] == 0].plot(color='lightgrey', ax=ax1, **{'edgecolor':'black'})
 
#Set figure options and plot 
fig.tight_layout()    
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=50) 
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
plt.show()

### Food Insecurity

In all four states, as percent food insecurity increases, percent low birth weight also increases. 
# # Discussion and Conclusion
The overall goal of this analysis was to provide clear insights into the relationships between nitrogen and low birth weight in missouri and surrounding states at the county level. Geospatial techniques, local models (GWR, MGWR), provided a quantitative representation of the determinants that may influence low birth weight rates.

The key findings from this research were that a set of sociodemographic and health variables were found to impact county low birth weight rates and that these factors vary geographically.

### Nitrogen

More specifically, our results indicated considerable spatial heterogeneity in low birth weight rates along the low to high nitrogen rate gradient. In parts of Iowa and west Nebraska, as nitrogen rates increase, rates of low birth weight decrease. Contrastingly, in Kansas, central Nebraska and northern Missouri as nitrogen rates increase so does low birth weight.   

These findings indicate that the impact of nitrogen rate (amount of nitrogen from fertilizer divided by amount of cropland treated in each county) varies by county. This could be due to factors in each county such as how they deal with nitrogen run-off, types of water safety measures in place, crop rotation and other agricultural practices that could affect how much nitrogen polutes drinking water.

### Rural

In Iowa, Kansas, eastern Nebraska, and north west Missouri, as percent rural population increases, rates of low birth weight decrease. In south central Nebraska as percent rural population increases so does low birth weight.  

### Preventable Hospital Stays

In Missouri, west Iowa, east Nebraska and east Kansas, as percentage of preventable hospital stays increases so does percent low birth weight.

### Physical Inactivity

In Iowa, Kansas, southern Missouri and east Nebraska as percentage of physical inactivity increases so does percent low birth weight. In central south Nebraska, as percentage of physical inactivity increases low birth weight decreases. Hot spots and cold spots for physical inactivity and low birth weight overlap in Iowa and Missouri.

### Percent African American

The findings demonstrated that the african american population was an influential factor in explaining spatial variation in low birth weight particulary, in Iowa, Nebraska and west Kansas as percent African American icreases, percent low birth weight also increases. 

### Food Insecurity

In all four states, as percent food insecurity increases, percent low birth weight also increases. 
# In[ ]:


# Convert GeoPandas DataFrame to regular pandas DataFrame
dfmw_e = pd.DataFrame(dfmw)

dfmw_e = dfmw_e.drop('geometry', axis=1)

pd.set_option("max_columns", None)
pd.set_option("max_rows", None)


# In[ ]:


dfmw_e['zero_nitrogen'] = mgwr_filtered_t[:,1] #X1 - Nitrogen filtered t-vals

#If there are insignificnt parameters make them 0 instead of the parameter estimate

dfmw_e['mgwr_nitrogen_sig'] = np.where(
    dfmw_e['zero_nitrogen'] == 0.0,
    0,
    dfmw_e['mgwr_nitrogen']
)

dfmw_e


# In[ ]:


#Exporting data for PowerBI data story
dfmw_e.to_csv('lowbirthweight_mgwr.csv',index=False)

