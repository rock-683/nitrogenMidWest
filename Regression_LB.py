#!/usr/bin/env python
# coding: utf-8

# # OLS Regression: Low Birthweight
# Ordinary least squares regression will be used to evaluate strength of explanatory variables and to aid in the initial reduction of variables

# ## Low Birthweight

# In[ ]:


import pandas as pd
import geopandas as gpd  # Spatial data manipulation

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


import sys
import pandas as pd
import geopandas as gpd  # Spatial data manipulation
get_ipython().system('{sys.executable} -m pip install CensusData')
get_ipython().system('{sys.executable} -m pip install pandas-datapackage-reader')
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


# In[ ]:


county_data = pd.read_csv('NCH2017.csv')  # county_nitrogen + county_health


# In[ ]:


pd.set_option("max_columns", None)
pd.set_option("max_rows", None)
county_data.head(1)


# ## Variables based on forward selection
# 
# Filter dataset down to selected independent/predictor variables and the target health outcome: **LBW**

# In[ ]:



#subset of the columns with selected variable/variable names
df = county_data[['NitrogenRate 2017',
                  'Preventable hospital stays 2017 raw value',
                  'Physical inactivity 2017 raw value',
                  'percent Non-Hispanic African American2017  raw value',
                  'Food insecurity 2017 raw value',
                  'percent Rural 2010 raw value', 
                  '% Low Birthweight 20142020']]

#Simplifying column header names
df.columns = ['NitrogenRate', 'PrevHospitalStay','PhysicalInactivity',
             'AfricanAmerican','FoodInsecurity','PercentRural','LBW']

df.head(1)


# In[ ]:


#checking for missing values
df.isna().sum()


# In[ ]:


# Dropping rows with missing values
df = df.dropna()

#remaining county count
len(df)


# # Correlation Matrix

# In[ ]:


corr = df.corr().abs() # absolute value of Pearson's r

display(corr)


# In[ ]:


# these lines will create mask so we don't see the top triangle of the heatmap, which is a mirror of the bottom.
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# nicer colormap
cmap = sns.light_palette("#2ecc71", as_cmap=True)

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(111)

sns.heatmap(corr, ax=ax1, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, cmap=cmap)


# # Ordinary Least Squares (OLS)/Linear Regression
# 
# Ordinary Least Squares (OLS) regression to evaluate strength of explanatory variables and to aid in the reduction of variables. 

# In[ ]:


from sklearn.model_selection import train_test_split
import statsmodels.api as sm

datav3 = df[['NitrogenRate','PrevHospitalStay','PhysicalInactivity','AfricanAmerican',
            'FoodInsecurity','PercentRural']]

mod = sm.OLS(df['LBW'], sm.add_constant(datav3))
res = mod.fit()
print(res.summary())


# In[ ]:


datav5 = df[['NitrogenRate','PrevHospitalStay','PhysicalInactivity',
            'FoodInsecurity','PercentRural']]

mod = sm.OLS(df['LBW'], sm.add_constant(datav5))
res = mod.fit()
print(res.summary())


# In[ ]:


datav6 = df[['NitrogenRate','PrevHospitalStay','PhysicalInactivity','AfricanAmerican',
            'FoodInsecurity']]

mod = sm.OLS(df['LBW'], sm.add_constant(datav6))
res = mod.fit()
print(res.summary())


# In[ ]:


datav7 = df[['NitrogenRate','PrevHospitalStay','PhysicalInactivity',
            'FoodInsecurity']]

mod = sm.OLS(df['LBW'], sm.add_constant(datav7))
res = mod.fit()
print(res.summary())


# In[ ]:


datav4 = df[['NitrogenRate','PhysicalInactivity','FoodInsecurity']]

mod = sm.OLS(df['LBW'], sm.add_constant(datav4))
res = mod.fit()
print(res.summary())


# ## Best Model Fit
# 
# datav3, Adjusted R-squared: 0.321; 
# The six variables that provided the highest explanitory strength for low birthweight were the following independent/predictor variables: ['NitrogenRate','PrevHospitalStay','PhysicalInactivity','AfricanAmerican','FoodInsecurity','PercentRural']]
