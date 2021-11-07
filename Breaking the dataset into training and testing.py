#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
warnings.simplefilter('ignore')


# In[7]:


data = pd.read_csv(r'C:\Users\sumanth\Downloads\Credit-Card-Fraud-Detection-using-Machine-Learning-master\creditcard.csv')
df = data.copy() # To keep the data as backup
df.head()


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


df.dtypes


# In[11]:


df.Time.tail(15)


# In[12]:


df.describe()


# # Checking the frequency of frauds before moving forward

# In[13]:


df.Class.value_counts()


# In[14]:


sns.countplot(x=df.Class, hue=df.Class)


# # Checking the distribution of amount

# In[15]:


plt.figure(figsize=(10, 5))
sns.distplot(df.Amount)


# In[16]:


df['Amount-Bins'] = ''


# Since, it is a little difficult to see. Let's engineer a new feature of bins.

# In[17]:


def make_bins(predictor, size=50):
    '''
    Takes the predictor (a series or a dataframe of single predictor) and size of bins
    Returns bins and bin labels
    '''
    bins = np.linspace(predictor.min(), predictor.max(), num=size)

    bin_labels = []

    # Index of the final element in bins list
    bins_last_index = bins.shape[0] - 1

    for id, val in enumerate(bins):
        if id == bins_last_index:
            continue
        val_to_put = str(int(bins[id])) + ' to ' + str(int(bins[id + 1]))
        bin_labels.append(val_to_put)
    
    return bins, bin_labels


# In[18]:


bins, bin_labels = make_bins(df.Amount, size=10)


# Now, adding bins in the column Amount-Bins.

# In[19]:


df['Amount-Bins'] = pd.cut(df.Amount, bins=bins,
                           labels=bin_labels, include_lowest=True)
df['Amount-Bins'].head().to_frame()


# Let's plot the bins.

# In[20]:


df['Amount-Bins'].value_counts()


# In[21]:


plt.figure(figsize=(15, 10))
sns.countplot(x='Amount-Bins', data=df)
plt.xticks(rotation=45)


# Since, count of values of Bins other than '0 to 2854' are difficult to view. Let's not insert the first one.

# In[22]:


plt.figure(figsize=(15, 10))
sns.countplot(x='Amount-Bins', data=df[~(df['Amount-Bins'] == '0 to 2854')])
plt.xticks(rotation=45)


# We can see that mostly the amount is between 0 and 2854 euros.

# In[ ]:




