#!/usr/bin/env python
# coding: utf-8

# # Correlation 

# Correlations between Low Birthweight, other county health indicators and Nitrogen. 
# <br>Scope: Midwest

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('NCH2017.csv') # county_nitrogen + county_health


# In[ ]:


# see all columns

print(df.columns.tolist())


# ## Correlations with Low Birthweight 

# In[ ]:


# selected variables
selectedvariables1 = df[['NitrogenRate 1997', 'NitrogenRate 2002','NitrogenRate 2007',
                           'NitrogenRate 2012', 'NitrogenRate 2017',
                          '% Low Birthweight 20142020']]


# In[ ]:


# Nitrogen

corr = selectedvariables1.corr()
corr2 = corr.iloc[-1:, :-1]
corr2


# In[ ]:


plt.figure(figsize=(16,2))
sns.heatmap(data = corr2, annot = True, annot_kws={"size":14},cmap="winter")
ax = plt.xticks(fontsize =14)
plt.yticks(fontsize =14)

plt.show()


# <br>NitrogenRate 2017 = 0.183359

# In[ ]:


# selected variables

selectedvariables2 = df[[

  # Demographics    
 'percent Rural 2010 raw value',
 'Population 2017 raw value',
 'percent below 18 years of age 2017 raw value',
 'percent 65 and older 2017 raw value',
 'percent Non-Hispanic African American2017  raw value',
 'percent American Indian and Alaskan Native 2017 raw value',
 'percent Asian 2017 raw value',
 'percent Native Hawaiian/Other Pacific Islander 2017 raw value',
 'percent Hispanic 2017 raw value',
 'percent Non-Hispanic white 2017 raw value',
 'percent not proficient in English 20132017 raw value',
 'percent Females 2017 raw value',
 '% Low Birthweight 20142020'
    
                      ]] 


# In[ ]:


# Demographics

corr = selectedvariables2.corr()
corr2 = corr.iloc[-1:, :-1]
corr2


# In[ ]:


plt.figure(figsize=(16,2))
sns.heatmap(data = corr2, annot = True, annot_kws={"size":14},cmap="winter")
ax = plt.xticks(fontsize =14)
plt.yticks(fontsize =14)
plt.show()


# <br> percent Rural 2010 raw value = -0.133786
# <br> percent Non-Hispanic African American 2017 raw value = 0.364896

# In[ ]:


selectedvariables3 = df[[
  
      
  # Access to Care
 'Uninsured 2017 raw value', # Access to Care
 'Primary care physicians 2017 raw value', # Access to Care
 'Ratio of population to primary care physicians 2017', # Access to Care
 'Uninsured adults 2017 raw value', # Access to Care
 'Uninsured children 2017 raw value',# Access to Care
 'Mental health providers 2017 raw value', # access to care
 'Ratio of population to mental health 2017 providers', #access to care
 'Other primary care providers 2017 raw value',# access to care
 'Ratio of population to primary care providers other than physicians 2017', # access to care
 'Dentists 2017 raw value','Ratio of population to 2017 dentists.', #access to care
 'Ratio of population to primary care physicians 2017', #access to care
 '% Low Birthweight 20142020'   
    ]] 


# In[ ]:


# Access to Care

corr = selectedvariables3.corr()
corr2 = corr.iloc[-1:, :-1]
corr2


# In[ ]:


plt.figure(figsize=(16,2))
sns.heatmap(data = corr2, annot = True, annot_kws={"size":14},cmap="winter")
ax = plt.xticks(fontsize =14)
plt.yticks(fontsize =14)
plt.show()


# In[ ]:


selectedvariables4 = df[[
                     
  # Air and Water Quality 
 'Drinking water violations 2018 raw value', # Air and Water Quality 
 'Air pollution - particulate matter 2016 raw value', # Air and Water Quality
 'Drinking water violations 2017 raw value', # Air and Water Quality  
    
  #Quality of Care  
 'Preventable hospital stays 2017 raw value',  # Quality of Care
 'Mammography screening 2017 raw value', # Quality of Care
 '% Low Birthweight 20142020' 
   
    ]] 


# In[ ]:


corr = selectedvariables4.corr()
corr2 = corr.iloc[-1:, :-1]
corr2


# In[ ]:


plt.figure(figsize=(16,2))
sns.heatmap(data = corr2, annot = True, annot_kws={"size":14},cmap="winter")
ax = plt.xticks(fontsize =14)
plt.yticks(fontsize =14)
plt.show()


# In[ ]:


selectedvariables5 = df[[
  
  # Health Outcomes
 'Premature death 20152017 raw value', # Health Outcomes
 'Premature death 2016-2018 raw value', # Health Outcomes
 'Poor or fair health 2017 raw value', # Health Outcomes
 'Poor physical health days 2017 raw value', # Health Outcomes
 'Poor mental health days 2017 raw value', # Health Outcomes
 'Adult smoking 2017 raw value', # Health Behaviors
 'Insufficient sleep 2016 raw value', # Health Behaviors
 '% Low Birthweight 20142020'
   
    ]] 


# In[ ]:



corr = selectedvariables5.corr()
corr2 = corr.iloc[-1:, :-1]
corr2


# In[ ]:


selectedvariables6 = df[[

  #Diet and Exercise
 'Food environment 20152017 index raw value', # Diet and Exercise
 'Physical inactivity 2016 raw value', # Diet and Exercise
 'Food insecurity 2017 raw value',  # Diet and Exercise
 'Limited access to healthy foods 2015 raw value', # Diet and Exercise
 'Adult obesity 2017 raw value', # Diet and Exercise
 'Physical inactivity 2017 raw value', # Diet and Exercise
 'Access to exercise 20102018 opportunities raw value', # Diet and Exercise 
    
      # Economic Factors
 'Median household income 2017 raw value', # Economic Factors
 'Severe housing cost burden 20132017 raw value', # Economic Factors 
    
 '% Low Birthweight 20142020'  
    ]] 


# In[ ]:



corr = selectedvariables6.corr()
corr2 = corr.iloc[-1:, :-1]
corr2


# In[ ]:


plt.figure(figsize=(16,2))
sns.heatmap(data = corr2, annot = True, annot_kws={"size":14},cmap="winter")
ax = plt.xticks(fontsize =14)
plt.yticks(fontsize =14)
plt.show()


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
