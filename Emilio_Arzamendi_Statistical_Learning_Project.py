#!/usr/bin/env python
# coding: utf-8

# Emilio Arzamendi

# In[1]:


# 1. Import libraries
import pandas as pd
import numpy as np
from scipy import stats


# In[2]:


# 2. Load csv dataset
df_costs = pd.read_csv("C:\\Users\\Emilio\\Desktop\\PGP\\Python\\insurance.csv")
print(df_costs)


# In[ ]:





# In[3]:


# 3a. Data shape and type
type(df_costs)


# In[4]:


df_costs.shape


# In[5]:


# 3b. Column types and names
df_costs.info()


# In[6]:


df_costs.columns


# In[7]:


# 3c. Missing values
df_costs.isna().sum()


# In[8]:


# 3d. 5 Point Summary
np.mean(df_costs, axis = 0)
df_costs[["age", "bmi", "children", "charges"]].describe()


# In[9]:


# 3e. Numerical Data Distribution
df_hist = df_costs.drop(columns=["sex", "children", "smoker", "region"])
df_hist.hist()


# In[10]:


# 3f. Skewness
df_costs[["age", "bmi", "children", "charges"]].skew(axis=0)


# In[11]:


# 3g. Check for outliers, first array is row number, second array column number, where z value is larger than 3 in each respective column
z = np.abs(stats.zscore(df_costs[["age", "bmi", "charges"]]))
z3 = np.where(z > 3)
z3


# In[12]:


# Alternatively we can create a matrix where TRUE if z > 3
zsum = z > 3
zsum = pd.DataFrame(zsum)
zsum.columns = ["age", "bmi", "charges"]
zsum
for i in ["age", "bmi", "charges"]:
 print(zsum[i].value_counts())


# In[13]:


# 3h. Categorical data distribution
for i in ["sex", "children", "smoker", "region"]:
 print(df_costs[i].value_counts())


# In[14]:


# 3i. Pair plot
pd.plotting.scatter_matrix(df_costs, alpha=0.2)


# In[15]:


# 4.a Do charges of people who smoke differ significantly
# from the people who don't?


# In[16]:


t_test = df_costs[["smoker", "charges"]]
t_test
t_test['smoker']


# In[17]:


groupY_smk = np.array(t_test.where(t_test.smoker == "yes").dropna()['charges'])


# In[18]:


groupN_smk = np.array(t_test.where(t_test.smoker == "no").dropna()['charges'])


# In[19]:


stats.ttest_ind(groupY_smk, groupN_smk)


# In[20]:


xs = t_test.loc[t_test['smoker'] == 'yes'].drop(columns=['smoker'])


# In[21]:


ysmk =[]
nsmok = []
for i in t_test.index:
    if t_test.iloc[i][0] == 'yes':
        ysmk.append(t_test.iloc[i][1])
    else:
        nsmok.append(t_test.iloc[i][1])
        


# In[22]:


seriessmk = pd.Series(ysmk)
seriesn = pd.Series(nsmok)


# In[23]:


seriessmk


# In[24]:


seriessmk.mean()


# In[25]:


stats.ttest_ind(xs, seriesn)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics


# In[26]:


df_costs


# In[27]:


df_costs_features = df_costs.drop(columns='charges')
y = df_costs['charges']
df_costs_features


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(df_costs_features, y, test_size=0.33, random_state=42)
X_train = pd.get_dummies(X_train) 
X_test = pd.get_dummies(X_test) 
X_train.shape


# In[29]:


clf = RandomForestRegressor(max_depth=3, n_estimators =5)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[30]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




