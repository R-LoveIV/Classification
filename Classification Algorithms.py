#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


#load the Dataset
mnist = fetch_openml('mnist_784')


# In[3]:


#Explode the data set
mnist


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import matplotlib
import matplotlib.pyplot as plt


# In[6]:


#seperate into two varriables
x, y = mnist['data'], mnist['target']


# In[15]:


random_digit = x[3600]
some_random_digit = random_digit.reshape(28,28)
plt.imshow(some_random_digit,cmap=matplotlib.cm.binary, interpolation="nearest")


# In[18]:


x_train, x_test = x[:6000] , x[6000:7000]
y_train , y_test = y[:6000], y[6000:7000]


# In[9]:


import numpy as np

shuffle_index = np.random.permutation(6000)
x_train , y_train = x_train[shuffle_index], y_train[shuffle_index]


# In[23]:


y_train_2 = y_train.astype(np.int8)
y_test_2 = y_test.astype(np.int8)
y_train_3 = (y_train == 2)
y_test_3 = (y_test == 2)


# In[25]:


y_test_2


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


clf = LogisticRegression(tol=0.1)


# In[28]:


clf.fit(x_train,y_train_2)


# In[29]:


y_pred = clf.predict([random_digit])


# In[30]:


y_pred


# In[31]:


from sklearn.model_selection import cross_val_score


# In[32]:


a = cross_val_score(clf,x_train,y_train_2,cv=3,scoring="accuracy")


# In[33]:


a.mean()


# In[ ]:




