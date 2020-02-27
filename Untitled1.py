
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
font = {'family':"IPAexGothic",
        'weight' : 'bold'}
mpl.rc('font', **font)


# In[2]:


X_train = pd.read_csv("./preprocessed_data/preprocessed_train_1.csv")
X_test = pd.read_csv("./preprocessed_data/preprocessed_test_1.csv")
orig_columns = X_train.columns.values


# # Model

# In[3]:


y_train = X_train["賃料"]
X_train = X_train.drop(columns=["賃料", "id"])


# In[4]:


X_test = X_test.drop(columns="id")


# In[5]:


features = list(X_train.columns)


# In[6]:


X_train.shape


# In[7]:


X_test.shape


# In[8]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))


# In[9]:


imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


# In[10]:


scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[11]:


print('Training data shape: ', X_train.shape)
print('Testing data shape: ', X_test.shape)


# In[12]:


import xgboost as xgb
model = xgb.XGBRegressor(n_jobs=-1,
                         max_depth=10,
                         n_estimators=1700,
                         learning_rate=0.05,
                         random_state=0,
                         gamma=0.3,
                         subsample=0.8,
                        reg_lambda=4,
                        eval_metric="rmse",
                        tree_method="gpu_hist",
                        verbose=-1)


# In[13]:


from sklearn.model_selection import cross_val_score


# In[ ]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tqdm import tqdm
import pickle

gbms = []
for i in range(50):
    gbm = xgb.XGBRegressor(n_estimators=50,
                            n_jobs=-1,
                            num_leaves=int(2**6.18),
                            feature_fraction=.63,
                            lambda_l1=10**-2.68,
                            lambda_l2=10**-2.67,
                            min_data_in_leaf=3,
                            learning_rate=10**-1.46,
                            num_boost_round=1000,
                            random_state=2434 + i,
                          tree_method="gpu_hist")

    gbm.fit(X_train, y_train)
    # test_pred = gbm.predict(test_X) * test_X["capacity"].values
    # smpsb_df.iloc[:len(test_X), 1] += test_pred / 50
    gbms.append(gbm)

