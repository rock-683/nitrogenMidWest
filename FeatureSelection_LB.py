#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: lightblue;">On occasion the container/server may need to be reset before this notebook can run, depending on which notebooks were run before it</div>

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
import statsmodels.api as sm


# In[ ]:


county_data = pd.read_csv('NCH2017.csv') # county_nitrogen + county_health


# The county_health table contains about **85 predictor variables** with **1 target variable.**: Low Birthweight 
# 
# Columns that are correlated with low birthweight by at least r=.3 - with the exception of nitrogen rate and rural which will be added regarless of r value - will be used. See Correlation_LB notebook for full correlations. Results are below. 

# Correlated with Low Birthweight by at least r =.3 
# <br>*`NitrogenRate 2017` and `percent Rural 2010 raw value` added obligatorily* 
# 
# <br> **`NitrogenRate 2017` = 0.183359**
# <br> `percent Rural 2010 raw value` = -0.133786
# <br> `Physical inactivity 2016 raw value` = 0.306668
# <br> `Preventable hospital stays 2017 raw value` = 0.313979
# <br> `Severe housing cost burden 20132017 raw value` = 0.318584
# <br> `Food environment 20152017 index raw value` = -0.323875
# <br> `Median household income 2017 raw value` = -0.34245
# <br> `Physical inactivity 2017 raw value` = 0.353853
# <br> `percent Non-Hispanic African American2017 raw value` = 0.364896
# <br> `Food insecurity 2017 raw value` = 0.449392

# In[ ]:


# Trying feature selection process once with Nitrogen cropland (cl) and once with Nitrogen landarea (la)


# In[ ]:


#Filter 
df = county_data[['NitrogenRate 2017','percent Rural 2010 raw value', 'Preventable hospital stays 2017 raw value',
            'Severe housing cost burden 20132017 raw value','Food environment 20152017 index raw value',
            'Median household income 2017 raw value','Physical inactivity 2017 raw value',
            'percent Non-Hispanic African American2017  raw value','Food insecurity 2017 raw value',
            'Mammography screening 2017 raw value','Uninsured adults 2017 raw value',
            'percent below 18 years of age 2017 raw value','Uninsured 2017 raw value',
            '% Low Birthweight 20142020']]


# In[ ]:


#checking for missing values
df.isna().sum()/len(county_data)


# In[ ]:


# what we started with
len(df)


# ### Drop missing values to run feature selection

# In[ ]:


# drop rows with missing values
df = df.dropna()

#how many are left
len(df)


# ## Wrapper methods
# 
# Within wrapper methods, there are different strategies for selecting the features. Namely, **forward selection**, **backward elimination** and **recursive feature elimination** strategies.
# 
# ### Forward selection
# 
# Forward selection is an iterative method that starts with having no feature in the model. 
# In each iteration, it keeps adding the feature which best improves the model.

# In[ ]:


#shuffle
dataset = df.sample(frac = 1).reset_index(drop=True)


# In[ ]:


# Store features and labels into variables X and y respectively.
X = dataset.iloc[:, :-1].to_numpy()
y = dataset['% Low Birthweight 20142020']


# In[ ]:


def forward_select(estimator, X, y, k=6):
    # this array holds indicators of whether each feature is currently selected
    selected = np.zeros(X.shape[1]).astype(bool)
    
    # fit and score model based on some subset of features
    score = lambda X_features: clone(estimator).fit(X_features, y).score(X_features, y)  # accurary is the measure
    
    # find indices to selected columns
    selected_indices = lambda: list(np.flatnonzero(selected))
    
    # repeat till k features are selected
    while np.sum(selected) < k:
        # indices to unselected columns
        rest_indices = list(np.flatnonzero(~selected))
    
        # compute model scores with an additional feature
        scores = [score(X[:, selected_indices() + [i]]) for i in rest_indices]
        print('\n%accuracy if adding column:\n   ',
              {i:int(s*100) for i,s in zip(rest_indices,scores)})
        
        # find index within `rest_indices` that points to the most predictive feature not yet selected 
        idx_to_add = rest_indices[np.argmax(scores)]
        print('add column', idx_to_add)
        
        # select this new feature
        selected[idx_to_add] = True
        
    return selected_indices()

support = sorted(forward_select(LinearRegression(), X, y))
#print(support)
print(forward_select(LinearRegression(), X, y))


# In[ ]:


fs_dataset = dataset.iloc[:, [0, 1, 2, 6, 7, 8]]

# features selected with forward selection
list(fs_dataset)


# In[ ]:


# Store features and labels into variables X and y respectively.
X = dataset.iloc[:, [0, 1, 2, 6, 7, 8]].to_numpy()
y = dataset['% Low Birthweight 20142020']

# create a linear regression model
model = LinearRegression()

# fit the model to the data
model.fit(X, y)

# calculate the R-squared of the model
r2 = model.score(X, y)

# print the R-squared
print("R-squared:", r2)

