#!/usr/bin/env python
# coding: utf-8

# In[102]:


#importing required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[103]:


#read csv file and check head

bike = pd.read_csv(r"day.csv")
bike.head()


# In[104]:


#check shape of data

bike.shape


# In[105]:


#check information

bike.info()


# In[106]:


#dropping unwanted columns
# As we have total as cnt, we can drop casual and registered
# Since we have days month and year separately, we don't require dteday
# instant is just a record number, so not required in analysis

bike = bike.drop(["casual","registered","dteday","instant"], axis=1)
bike.head()


# In[107]:


#Replacing numerical values with actual categories as given in dictionary for easy understanding of data and relations in EDA.

bike["season"] = bike["season"].map({1:'spring',2:'summer',3:'fall',4:'winter'})
bike["yr"] = bike["yr"].map({0:2018,1:2019})
bike["mnth"] = bike["mnth"].map({1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'})
bike["weekday"] = bike["weekday"].map({0:'sun',1:'mon',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat'})
bike["weathersit"] = bike["weathersit"].map({1:'clear',2:'misty',3:'rain',4:'heavy_rain'})
bike.head()


# In[108]:


# description of numerical values

bike.describe()


# In[109]:


#box plot with one parameter to check the data for outliers.

plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot(y='temp', data=bike)
plt.subplot(2,3,2)
sns.boxplot(y='atemp', data=bike)
plt.subplot(2,3,3)
sns.boxplot(y='hum', data=bike)
plt.subplot(2,3,4)
sns.boxplot(y='windspeed', data=bike)
plt.subplot(2,3,5)
sns.boxplot(y='cnt', data=bike)
plt.show()


# In[110]:


# There are not much outliers, we dont need to treat them and continue further with analysis


# In[111]:


#visualisation of data using pair plots

sns.pairplot(bike)
plt.show()


# In[112]:


plt.figure(figsize=(20,12))
plt.subplot(2,2,1)
sns.regplot(data=bike,y="cnt",x="temp")
plt.subplot(2,2,2)
sns.regplot(data=bike,y="cnt",x="atemp")
plt.subplot(2,2,3)
sns.regplot(data=bike,y="cnt",x="hum")
plt.subplot(2,2,4)
sns.regplot(data=bike,y="cnt",x="windspeed")
plt.show()


# In[113]:


# In pair plot we see that temp and atemp look more or less similar is data spread and pattern. 
# Demand is directly proportional to temperature.
# Demand is inversely proprtional to humidity and windspeed.


# In[114]:


# Creating boxplots to understand relation of cnt with other parameters

plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
sns.boxplot(x='yr', y='cnt', data=bike)
plt.subplot(3,3,2)
sns.boxplot(x='mnth', y='cnt', data=bike)
plt.subplot(3,3,3)
sns.boxplot(x='holiday', y='cnt', data=bike)
plt.subplot(3,3,4)
sns.boxplot(x='weekday', y='cnt', data=bike)
plt.subplot(3,3,5)
sns.boxplot(x='workingday', y='cnt', data=bike)
plt.subplot(3,3,6)
sns.boxplot(x='weathersit', y='cnt', data=bike)
plt.subplot(3,3,7)
sns.boxplot(x='season', y='cnt', data=bike)
plt.show()


# In[115]:


# Demand was higher in 2019 than 2018.
# Demand takes a peak around july and then starts dropping. First quarter of year stays slow.
# Demand is on lower side on holidays.
# Medians are almost same for all days of week, be it a working day or not.
# On rainy days the rentals counts are very low.
# Demand is less during spring.


# In[116]:


# Creating dummy variables for regression

bike.head()


# In[117]:


seadum = pd.get_dummies(bike["season"], drop_first=True)
mondum = pd.get_dummies(bike["mnth"], drop_first=True)
weekdum = pd.get_dummies(bike["weekday"], drop_first=True)
weatdum = pd.get_dummies(bike["weathersit"], drop_first=True)

#For weathersit already there is one category less but to accommodate future data we are not dropping a column


# In[118]:


# Merging the dummy colums with actual data

bike = pd.concat([bike,seadum,mondum,weekdum,weatdum], axis=1)
bike.head()


# In[119]:


#Since dummies are added, we will drop original colums

bike = bike.drop(['season','mnth','weekday','weathersit'], axis=1)

#Substituting year with 0 and 1

bike["yr"] = bike["yr"].map({2018:0,2019:1})

bike.head()


# In[120]:


# Splitting data into train and test
bike_train, bike_test = train_test_split(bike, train_size=0.70, random_state=100)
print(bike_train.shape)
print(bike_test.shape)


# In[121]:


# Rescaling of columns like temp, hum etc using normalisation
scaler = MinMaxScaler()

num_var =['temp','atemp','hum','windspeed','cnt']

bike_train[num_var]= scaler.fit_transform(bike_train[num_var])

bike_train.head()


# In[122]:


bike_train[num_var].describe()


# In[123]:


# TRAINING THE MODEL

#Plotting heatmap to check the correlation of independant variables

plt.figure(figsize=(25,15))
sns.heatmap(bike_train.corr(), annot=True, cmap="YlGnBu")
plt.show()


# In[124]:


# X-train and y_train
y_train = bike_train.pop('cnt')
X_train = bike_train


# In[125]:


X_train.columns


# In[126]:


# Using RFE to minimise columns
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[127]:


#Reduce columns to 15 using RFE
rfe = RFE(estimator=lr,n_features_to_select=15)
rfe.fit(X_train,y_train)



# In[128]:


X_train.columns[rfe.support_]


# In[129]:


X_train_15 = X_train[['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'summer',
       'winter', 'feb', 'jan', 'sep', 'sat', 'sun', 'misty', 'rain']]


# In[130]:


# Building model with all variables

# Model 1

X_train_15_sm = sm.add_constant(X_train_15)

# Creating a model
lr = sm.OLS(y_train, X_train_15_sm)

# Fit the model
lr_model = lr.fit()

# parameters
lr_model.summary()


# In[131]:


# We will drop a parameter based on p-value and VIF in sequence as follow
#high p-value and high vif
#high p-value and low vif
#low p-value and high vif

vif = pd.DataFrame()
vif['features'] = X_train_15.columns
vif['vif'] = [variance_inflation_factor(X_train_15.values, i) for i in range(X_train_15.shape[1])]
vif['vif']=round(vif['vif'],2)
vif=vif.sort_values(by="vif", ascending = False)
vif


# In[132]:


# We will drop holiday as it has high p-value

# Model 2

X = X_train_15.drop('holiday', axis=1)

#Rebuild model after dropping parameter

X_train_15_sm = sm.add_constant(X)

# Creating a model
lr = sm.OLS(y_train, X_train_15_sm)

# Fit the model
lr_model = lr.fit()

# parameters
lr_model.summary()


# In[133]:


vif = pd.DataFrame()
vif['features'] = X.columns
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['vif']=round(vif['vif'],2)
vif=vif.sort_values(by="vif", ascending = False)
vif


# In[134]:


# We will drop feb as it has high vif

# Model 3

X = X.drop('feb', axis=1)

#Rebuild model after dropping parameter

X_train_15_sm = sm.add_constant(X)

# Creating a model
lr = sm.OLS(y_train, X_train_15_sm)

# Fit the model
lr_model = lr.fit()

# parameters
lr_model.summary()


# In[135]:


vif = pd.DataFrame()
vif['features'] = X.columns
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['vif']=round(vif['vif'],2)
vif=vif.sort_values(by="vif", ascending = False)
vif


# In[136]:


# We will drop sun as it has high vif

# Model 4

X = X.drop('sun', axis=1)

#Rebuild model after dropping parameter

X_train_15_sm = sm.add_constant(X)

# Creating a model
lr = sm.OLS(y_train, X_train_15_sm)

# Fit the model
lr_model = lr.fit()

# parameters
lr_model.summary()


# In[137]:


vif = pd.DataFrame()
vif['features'] = X.columns
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['vif']=round(vif['vif'],2)
vif=vif.sort_values(by="vif", ascending = False)
vif


# In[138]:


# We will drop jan as it has high p-value

# Model 5

X = X.drop('jan', axis=1)

#Rebuild model after dropping parameter

X_train_15_sm = sm.add_constant(X)

# Creating a model
lr = sm.OLS(y_train, X_train_15_sm)

# Fit the model
lr_model = lr.fit()

# parameters
lr_model.summary()


# In[139]:


vif = pd.DataFrame()
vif['features'] = X.columns
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['vif']=round(vif['vif'],2)
vif=vif.sort_values(by="vif", ascending = False)
vif


# In[140]:


# We will drop hum as it has high vif

# Model 6

X = X.drop('hum', axis=1)

#Rebuild model after dropping parameter

X_train_15_sm = sm.add_constant(X)

# Creating a model
lr = sm.OLS(y_train, X_train_15_sm)

# Fit the model
lr_model = lr.fit()

# parameters
lr_model.summary()


# In[141]:


vif = pd.DataFrame()
vif['features'] = X.columns
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['vif']=round(vif['vif'],2)
vif=vif.sort_values(by="vif", ascending = False)
vif


# In[142]:


X.describe()


# In[143]:


# RESIDUAL ANALYSIS

y_train_pred = lr_model.predict(X_train_15_sm)


# In[144]:


res = y_train - y_train_pred
sns.displot(res)


# In[145]:


# Checking multicollinearity between final 10 variables
plt.figure(figsize=(25,15))
sns.heatmap(X.corr(),annot = True, cmap="YlGnBu")
plt.show()


# In[146]:


# No variable is correlated


# In[147]:


# PREDICTION AND EVALUATION OF TEST SET

#Rescaling of test set
# Rescaling of columns like temp, hum etc using normalisation

num_var =['temp','atemp','hum','windspeed','cnt']

bike_test[num_var]= scaler.transform(bike_test[num_var])


bike_test.head()


# In[148]:


bike_test.describe()


# In[149]:


# X-test and y_test
y_test = bike_test.pop('cnt')
X_test = bike_test


# In[152]:


col_10 = X.columns
X_test = X_test[col_10]


# In[153]:


# Adding constant
X_test_sm = sm.add_constant(X_test)
X_test_sm.head()


# In[154]:


y_test_pred = lr_model.predict(X_test_sm)


# In[161]:


#evaluate

r2 = r2_score(y_true=y_test,y_pred=y_test_pred)
print(r2)


# In[162]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_test_pred)


# In[163]:


# Calculating Adjusted-R^2 value for the test dataset

adjusted_r2 = round(1-(1-r2)*(X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1),4)
print(adjusted_r2)


# In[168]:


lr_model.params


# In[ ]:


# The equation for the line is as follow

# cnt = 0.08 + 0.23yr + 0.06workingday + 0.55temp - 0.16windspeed + 0.09summer + 0.13winter + 0.1sep + 0.07sat - 0.08misty - 0.29rain


# In[ ]:


# Demand of bikes depends on year,working day, temperature, windspeed, summer, winter, september, saturday, misty condition, rainy season.

