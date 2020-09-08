#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # NumPy Exercises 
# 
# Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks, and then you'll be asked some more complicated questions.

# #### Import NumPy as np

# In[3]:


import numpy as np


# #### Create an array of 10 zeros 

# In[6]:


np.zeros(10)


# #### Create an array of 10 ones

# In[26]:


np.ones(10)


# #### Create an array of 10 fives

# In[25]:


np.ones(10) * 5


# #### Create an array of the integers from 10 to 50

# In[27]:


np.arange(10,51)


# #### Create an array of all the even integers from 10 to 50

# In[32]:


np.arange(10,51,2)


# In[ ]:





# #### Create a 3x3 matrix with values ranging from 0 to 8

# In[33]:


arr= np.arange(9)
arr.reshape(3,3)


# #### Create a 3x3 identity matrix

# In[36]:


np.eye(3)


# #### Use NumPy to generate a random number between 0 and 1

# In[37]:


np.random.rand(1)


# #### Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution

# In[39]:


np.random.randn(25)


# #### Create the following matrix:

# In[42]:


np.arange(1,101).reshape(10,10) / 100


# #### Create an array of 20 linearly spaced points between 0 and 1:

# In[43]:


np.linspace(0,1,20)


# ## Numpy Indexing and Selection
# 
# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:

# In[46]:


mat = np.arange(1,26).reshape(5,5)
mat


# In[48]:


mat[2:,1:]


# In[40]:





# In[49]:


mat[3,4]


# In[41]:





# In[52]:


mat[:3,1:2]


# In[42]:





# In[53]:


mat[4,:]


# In[46]:





# In[54]:


mat[3:5,:]


# In[49]:





# ### Now do the following

# #### Get the sum of all the values in mat

# In[56]:


np.sum(mat)


# #### Get the standard deviation of the values in mat

# In[57]:


np.std(mat)


# #### Get the sum of all the columns in mat

# In[58]:


mat.sum(axis=0)


# # Great Job!
