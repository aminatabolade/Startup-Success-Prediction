#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import missingno

# Models from Scikit-Learn
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

#Saving and loading model
from joblib import dump, load


# In[2]:


ssp = pd.read_csv("startup data.csv")
ssp


# In[3]:


ssp.columns


# In[4]:


#dropping non-labeled and unneeded columns
ssp = ssp.drop(['Unnamed: 0', 'state_code', 'Unnamed: 6', 'latitude','longitude', 'zip_code','state_code.1','object_id'], axis=1 )


# In[5]:


ssp


# In[6]:


ssp.head()


# In[7]:


ssp.columns


# In[8]:


ssp.info()


# In[9]:


ssp.shape, len(ssp)


# In[10]:


ssp.isna().sum()


# In[11]:


#visualizing null data
missingno.matrix(ssp);


# In[12]:


#Visualizing data based on the age of first funding, last funding, first milestone nd last milestone
sns.set_style("ticks")
sns.pairplot(ssp[['first_funding_at', 'last_funding_at', 'age_first_funding_year',
       'age_last_funding_year', 'age_first_milestone_year',
       'age_last_milestone_year',"status"]],hue = "status", diag_kind = "kde",kind = "scatter",palette = "husl")
plt.show()


# In[13]:


#Visualizing based on startup going through roundA, roundB and roundC funding
sns.set_style("ticks")
sns.pairplot(ssp[['has_roundA', 'has_roundB', 'has_roundC', 'has_roundD',"status"]],hue = "status", diag_kind = "kde",kind = "scatter",palette = "husl")
plt.show()


# In[14]:


#Visualizing data based on whether startup has vc and angel;
sns.set_style("ticks")
sns.pairplot(ssp[['has_VC',
       'has_angel',"status"]],hue = "status", diag_kind = "kde",kind = "scatter",palette = "husl")
plt.show()


# In[15]:


#Visualizing based on Location
sns.set_style("ticks")
sns.pairplot(ssp[['is_CA', 'is_NY', 'is_MA', 'is_TX',
       'is_otherstate',"status"]],hue = "status", diag_kind = "kde",kind = "scatter",palette = "husl")
plt.show()


# In[16]:


#Visualizing based on product category
sns.set_style("ticks")
sns.pairplot(ssp[['is_software', 'is_web', 'is_mobile',
       'is_enterprise', 'is_advertising', 'is_gamesvideo', 'is_ecommerce',
       'is_biotech', 'is_consulting', 'is_othercategory',"status"]],hue = "status", diag_kind = "kde",palette = "husl")
plt.show()


# ## Machine Learning Model

# In[17]:


# Using the transformer pipeline

#filling null data
categorical_features = ['closed_at']
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

numeric_features = ['age_first_milestone_year', 'age_last_milestone_year']
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

#Convert to non-numerical data to numerical data
preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", categorical_transformer, categorical_features),
                        ("num", numeric_transformer, numeric_features)
                    ])

# Creating a preprocessing and modelling pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("model", RandomForestClassifier())])

# Split data into x and y
x = ssp.drop("status", axis=1)
y = ssp["status"]


# In[18]:


#split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[19]:


model.fit(x_train, y_train)


# In[20]:


model.score(x_train, y_train)


# In[21]:


model.score(x_test, y_test)


# In[22]:


y_preds = model.predict(x_test)
y_preds


# In[23]:


#Evaluating model using confusion matrix
confusion_matrix (y_test, y_preds)


# In[24]:


#Visualize confusion matrix with pd.crosstab()
pd.crosstab(y_test,
           y_preds,
           rownames=["Actual Labels"],
           colnames=["Predicted Labels"])


# In[25]:


plot_confusion_matrix(model, x, y);


# In[26]:


# Classification report of Model
print(classification_report(y_test, y_preds))


# In[27]:


# Save model to file
dump(model, filename="ssp_random_forest_model.joblib")

