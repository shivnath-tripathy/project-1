
# coding: utf-8

# In[3]:


import sqlite3
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from math import sqrt
from tqdm import tqdm_notebook
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


# In[4]:


# Create your connection.
cnx = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


# In[5]:


df.head() # Look at the sample of the data..


# In[6]:


df.describe().T


# In[7]:


print("Dimension of the Data \nRows: %d \nColumns: %d " %(df.shape[0], df.shape[1]))#Get Number of Rows and columns


# In[8]:


#Take Copy of the Data to Work On
inputData = df.copy(deep=True)


# In[9]:


# Check how many columns in a row has null value and take the count of it for each row
inputData.isnull().sum(axis=1).sort_values(ascending=False)[:15] 
# From the below its clear that, there are many rows which has 38 out of 42 columns null


# In[10]:


# Adding a new column with count of null values in a row
inputData['nancount'] =inputData.apply(lambda x: x.isnull().sum(), axis=1)


# In[11]:


# Group by to get number of rows based on the null count
inputData.groupby(['nancount']).count().iloc[:,0:1]
# From the results below it is identified that there are 836 rows where 38 fields are null, which can be removed


# In[12]:


inputData.isnull().sum(axis=0).sort_values(ascending=False)[:20]
# from below we can confirm that, there is no significant colum where more (may be 80%) of the values is null
# So this can be retained


# In[13]:


# Get count of Unique values in each columns
featureCounts = inputData.nunique(dropna = False)


# In[14]:


featureCounts.sort_values()
# Below results states that no columns has anything which is constant, if there was any we can remove it


# In[15]:


#Drop all rows where  no of columns which has null values is greater than 30
inputData.drop(inputData[(inputData ['nancount']>30)].index, inplace=True)


# In[16]:


#check if any rows in Target has a Null value?
inputData['overall_rating'].isnull().sum(axis=0)
#if available delete those rows 


# In[17]:


# Delete the 'nancount' column as it is no more required now
del inputData['nancount']
inputData.shape
#Another way to delete column
#inputData.drop('nancount', axis=1, inplace=True)


# In[18]:


#Get Remaining number of rows atleast one null value is there. 
#(Not to be confused with rows more than 30 only to be deleted.. That step can be removed and only below step can be used)
#Null values does not gives significant results so can be removed... predict only with clean data
inputData[inputData.isnull().sum(axis=1)>0].shape


# In[19]:


# Remove all na values as it may cause in mis-interpretion of data
inputData = inputData.dropna()
inputData.shape


# In[20]:


# Display All Column Names
inputData.columns 


# In[21]:


# Delete the columns which are irrelavant for the model predications
inputData.drop(['id','player_fifa_api_id','player_api_id','date'], axis=1, inplace=True)


# In[22]:


# Display Column Names
inputData.columns


# In[23]:


#Take copy of all the Object data type columns to a new dataframe - Done this for the purpose of encoding the categorical columns
obj_inputData = inputData.select_dtypes(include=['object']).copy()


# In[24]:


obj_inputData.columns #Check all categorical column names


# In[25]:


# Apply encoding using Factorize Method
obj_inputData =obj_inputData.apply(lambda x: pd.factorize(x)[0])


# In[26]:


obj_inputData.head()


# In[27]:


# Take copy of all numerical Data type to a new Dataframe
num_inputData = inputData.select_dtypes(exclude=['object']).copy()
num_inputData.head()


# In[28]:


#concatenate numerical dataframe and object dataframe
finalData = pd.concat([num_inputData, obj_inputData], axis=1)
finalData.head()


# In[29]:


#Segregate Feauters and Targerts
#Here X = Features and Y = Target
X= finalData.iloc[:,1:]
Y = finalData.iloc[:,0:1]


# In[30]:


#Split the data for training and testing
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state =3)


# In[30]:


#Check Cross Validation r2 Score
LIN = LinearRegression()
cv_results = cross_val_score(LIN, X, Y, cv=5,scoring="r2")
cv_results.mean()


# In[31]:


#Train Model Using Linear Regression
LIN_model = LinearRegression()
LIN_model.fit(X_train, Y_train)
# y_pred = LIN.predict(X_test)


# In[32]:


#Predic using model for the trained data
pred_train = LIN_model.predict(X_train)


# In[33]:


#Predict with Test Data for the Model and Check the MSE value
pred_test = LIN_model.predict(X_test)
print("MSE", mean_squared_error(Y_test, pred_test))


# In[34]:



print("Accuracy of the test data", LIN_model.score(X_test,Y_test)*100,"%")


# In[102]:


resudials_train = pred_train[:1000]- Y_train[:1000]
resudials_train['overall_rating'].max()


# In[103]:


resudials_test = pred_test[:1000]- Y_test[:1000]
resudials_test['overall_rating'].max()


# In[107]:


get_ipython().magic('matplotlib inline')
plt.figure(figsize=(10,8))
plt.plot(pred_train[:1000],resudials_train,'o',c="c",alpha=0.9, label = 'Train data')
# plt.plot(pred_test[:1000],resudials,'^',c="b",alpha=.9,  label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 100, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')


plt.yticks(np.arange(resudials_train['overall_rating'].min(), resudials_train['overall_rating'].max()))

# Y label
plt.ylabel("Residuals")

## plot title
plt.title("Residual plot using Test Data for First 1000 records")

plt.show()


# In[108]:


get_ipython().magic('matplotlib inline')
plt.figure(figsize=(10,8))
#plt.scatter(pred_train,pred_train- Y_train,c="b",s=40,alpha=0.9, label = 'Train data')
plt.plot(pred_test[:1000],resudials_test,'^',c="b",alpha=.9,  label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 100, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')


plt.yticks(np.arange(resudials_test['overall_rating'].min(), resudials_test['overall_rating'].max()))

# Y label
plt.ylabel("Residuals")

## plot title
plt.title("Residual plot using Test Data for First 1000 records")

plt.show()


# In[32]:


Y_pred[:10] #Get first 10 Predicted Value


# In[33]:


Y_test[:10] #Get first 10 Actual Value


# In[33]:


import pickle
filename = 'SoccerLinModel_Final.sav'


# In[ ]:



# save the model to disk
pickle.dump(LIN_model, open(filename, 'wb'))


# In[34]:


# Load the model from the disk and predict the values
SoccerLin_model = pickle.load(open(filename, 'rb'))
result = SoccerLin_model.score(X_train, Y_train)
print("Training Accuracy:", result *100, "%")
result = SoccerLin_model.score(X_test, Y_test)
print("Test Accuracy:", result *100, "%")


# In[35]:


# Use Logistic regression for predication...
LON = LogisticRegression()
cv_results = cross_val_score(LON, X, Y, cv=5,scoring="r2")
cv_results.mean()
# Results does not look good.

