#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.pieriandata.com"><img src="../Pierian_Data_Logo.PNG"></a>
# <strong><center>Copyright by Pierian Data Inc.</center></strong> 
# <strong><center>Created by Jose Marcial Portilla.</center></strong>

# # Keras API Project Exercise
# 
# ## The Data
# 
# We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
# 
# ## NOTE: Do not download the full zip from the link! We provide a special version of this file that has some extra feature engineering for you to do. You won't be able to follow along with the original file!
# 
# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.
# 
# ### Our Goal
# 
# Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model thatcan predict wether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. Keep in mind classification metrics when evaluating the performance of your model!
# 
# The "loan_status" column contains our label.
# 
# ### Data Overview

# ----
# -----
# There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>LoanStatNew</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>loan_amnt</td>
#       <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>term</td>
#       <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>int_rate</td>
#       <td>Interest Rate on the loan</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>installment</td>
#       <td>The monthly payment owed by the borrower if the loan originates.</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>grade</td>
#       <td>LC assigned loan grade</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>sub_grade</td>
#       <td>LC assigned loan subgrade</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>emp_title</td>
#       <td>The job title supplied by the Borrower when applying for the loan.*</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>emp_length</td>
#       <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>home_ownership</td>
#       <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>annual_inc</td>
#       <td>The self-reported annual income provided by the borrower during registration.</td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td>verification_status</td>
#       <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td>issue_d</td>
#       <td>The month which the loan was funded</td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td>loan_status</td>
#       <td>Current status of the loan</td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td>purpose</td>
#       <td>A category provided by the borrower for the loan request.</td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td>title</td>
#       <td>The loan title provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td>zip_code</td>
#       <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td>addr_state</td>
#       <td>The state provided by the borrower in the loan application</td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td>dti</td>
#       <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>earliest_cr_line</td>
#       <td>The month the borrower's earliest reported credit line was opened</td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>open_acc</td>
#       <td>The number of open credit lines in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>20</th>
#       <td>pub_rec</td>
#       <td>Number of derogatory public records</td>
#     </tr>
#     <tr>
#       <th>21</th>
#       <td>revol_bal</td>
#       <td>Total credit revolving balance</td>
#     </tr>
#     <tr>
#       <th>22</th>
#       <td>revol_util</td>
#       <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
#     </tr>
#     <tr>
#       <th>23</th>
#       <td>total_acc</td>
#       <td>The total number of credit lines currently in the borrower's credit file</td>
#     </tr>
#     <tr>
#       <th>24</th>
#       <td>initial_list_status</td>
#       <td>The initial listing status of the loan. Possible values are – W, F</td>
#     </tr>
#     <tr>
#       <th>25</th>
#       <td>application_type</td>
#       <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
#     </tr>
#     <tr>
#       <th>26</th>
#       <td>mort_acc</td>
#       <td>Number of mortgage accounts.</td>
#     </tr>
#     <tr>
#       <th>27</th>
#       <td>pub_rec_bankruptcies</td>
#       <td>Number of public record bankruptcies</td>
#     </tr>
#   </tbody>
# </table>
# 
# ---
# ----

# ## Starter Code
# 
# #### Note: We also provide feature information on the data as a .csv file for easy lookup throughout the notebook:

# In[7]:


import pandas as pd


# In[8]:


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[9]:


print(data_info.loc['revol_util']['Description'])


# In[10]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[11]:


feat_info('mort_acc')


# ## Loading the data and other imports

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


df = pd.read_csv('../DATA/lending_club_loan_two.csv')


# In[14]:


df.info()


# # Project Tasks
# 
# **Complete the tasks below! Keep in mind is usually more than one way to complete the task! Enjoy**
# 
# -----
# ------
# 
# # Section 1: Exploratory Data Analysis
# 
# **OVERALL GOAL: Get an understanding for which variables are important, view summary statistics, and visualize the data**
# 
# 
# ----

# **TASK: Since we will be attempting to predict loan_status, create a countplot as shown below.**

# In[15]:


sns.countplot(data=df, x="loan_status")


# In[ ]:





# **TASK: Create a histogram of the loan_amnt column.**

# In[16]:


plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'], kde=False, bins=40)


# In[ ]:





# **TASK: Let's explore correlation between the continuous feature variables. Calculate the correlation between all continuous numeric variables using .corr() method.**

# In[17]:


df.corr()


# In[ ]:





# **TASK: Visualize this using a heatmap. Depending on your version of matplotlib, you may need to manually adjust the heatmap.**
# 
# * [Heatmap info](https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap)
# * [Help with resizing](https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot)

# In[18]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.ylim(10,0)


# In[ ]:





# **TASK: You should have noticed almost perfect correlation with the "installment" feature. Explore this feature further. Print out their descriptions and perform a scatterplot between them. Does this relationship make sense to you? Do you think there is duplicate information here?**

# In[19]:


feat_info('installment')


# In[20]:


feat_info('loan_amnt')


# In[ ]:





# In[ ]:





# In[21]:


sns.scatterplot(x='installment',y='loan_amnt', data=df)


# **TASK: Create a boxplot showing the relationship between the loan_status and the Loan Amount.**

# In[22]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# In[ ]:





# **TASK: Calculate the summary statistics for the loan amount, grouped by the loan_status.**

# In[23]:


df.groupby('loan_status')['loan_amnt'].describe()


# In[ ]:





# **TASK: Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans. What are the unique possible grades and subgrades?**

# In[24]:


df['grade'].unique()


# In[25]:


df['sub_grade'].unique()


# In[26]:


feat_info('sub_grade')


# **TASK: Create a countplot per grade. Set the hue to the loan_status label.**

# In[27]:


sns.countplot(x='grade',data=df,hue='loan_status')


# In[ ]:





# **TASK: Display a count plot per subgrade. You may need to resize for this plot and [reorder](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot) the x axis. Feel free to edit the color palette. Explore both all loans made per subgrade as well being separated based on the loan_status. After creating this plot, go ahead and create a similar plot, but set hue="loan_status"**

# In[ ]:





# In[28]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm')


# In[ ]:





# In[29]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm', 
              hue='loan_status')


# In[ ]:





# **TASK: It looks like F and G subgrades don't get paid back that often. Isloate those and recreate the countplot just for those subgrades.**

# In[30]:


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade', data=f_and_g, order=subgrade_order, palette='coolwarm', 
              hue='loan_status')


# In[ ]:





# **TASK: Create a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".**

# In[31]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off': 0})


# In[32]:


df[['loan_repaid', 'loan_status']]


# In[ ]:





# In[ ]:





# **CHALLENGE TASK: (Note this is hard, but can be done in one line!) Create a bar plot showing the correlation of the numeric features to the new loan_repaid column. [Helpful Link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html)**

# In[33]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# In[ ]:





# ---
# ---
# # Section 2: Data PreProcessing
# 
# **Section Goals: Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables.**
# 
# 

# In[34]:


df.head()


# # Missing Data
# 
# **Let's explore this missing data columns. We use a variety of factors to decide whether or not they would be useful, to see if we should keep, discard, or fill in the missing data.**

# **TASK: What is the length of the dataframe?**

# In[35]:


len(df)


# In[ ]:





# **TASK: Create a Series that displays the total count of missing values per column.**

# In[36]:


df.isnull().sum()


# In[ ]:





# **TASK: Convert this Series to be in term of percentage of the total DataFrame**

# In[37]:


100 * df.isnull().sum() / len(df)


# In[ ]:





# In[ ]:





# **TASK: Let's examine emp_title and emp_length to see whether it will be okay to drop them. Print out their feature information using the feat_info() function from the top of this notebook.**

# In[38]:


feat_info('emp_title')


# In[ ]:





# **TASK: How many unique employment job titles are there?**

# In[39]:


df['emp_title'].nunique()


# In[ ]:





# In[40]:


df['emp_title'].value_counts()


# **TASK: Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.**

# In[41]:


df = df.drop('emp_title', axis=1)


# In[ ]:





# **TASK: Create a count plot of the emp_length feature column. Challenge: Sort the order of the values.**

# In[42]:


sorted(df['emp_length'].dropna().unique())


# In[ ]:





# In[43]:


emp_length_order = [
 '< 1 year',
 '1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years'
]


# In[44]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length', data=df, order=emp_length_order,
             hue='loan_status')


# **TASK: Plot out the countplot with a hue separating Fully Paid vs Charged Off**

# In[45]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length', data=df, order=emp_length_order)


# In[ ]:





# **CHALLENGE TASK: This still doesn't really inform us if there is a strong relationship between employment length and being charged off, what we want is the percentage of charge offs per category. Essentially informing us what percent of people per employment category didn't pay back their loan. There are a multitude of ways to create this Series. Once you've created it, see if visualize it with a [bar plot](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html). This may be tricky, refer to solutions if you get stuck on creating this Series.**

# In[46]:


emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']


# In[47]:


emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']


# In[48]:


emp_len = emp_co / (emp_co + emp_fp)


# In[49]:


emp_len.plot(kind='bar')


# In[ ]:





# In[ ]:





# **TASK: Charge off rates are extremely similar across all employment lengths. Go ahead and drop the emp_length column.**

# In[50]:


df = df.drop('emp_length', axis=1)


# In[ ]:





# **TASK: Revisit the DataFrame to see what feature columns still have missing data.**

# In[51]:


df.isnull().sum()


# In[ ]:





# **TASK: Review the title column vs the purpose column. Is this repeated information?**

# In[52]:


df['purpose'].head(10)


# In[ ]:





# In[53]:


df['title'].head(10)


# **TASK: The title column is simply a string subcategory/description of the purpose column. Go ahead and drop the title column.**

# In[54]:


df = df.drop('title', axis=1)


# In[ ]:





# ---
# **NOTE: This is one of the hardest parts of the project! Refer to the solutions video if you need guidance, feel free to fill or drop the missing values of the mort_acc however you see fit! Here we're going with a very specific approach.**
# 
# 
# ---
# **TASK: Find out what the mort_acc feature represents**

# In[55]:


feat_info('mort_acc')


# In[ ]:





# **TASK: Create a value_counts of the mort_acc column.**

# In[56]:


df['mort_acc'].value_counts()


# In[ ]:





# **TASK: There are many ways we could deal with this missing data. We could attempt to build a simple model to fill it in, such as a linear model, we could just fill it in based on the mean of the other columns, or you could even bin the columns into categories and then set NaN as its own category. There is no 100% correct approach! Let's review the other columsn to see which most highly correlates to mort_acc**

# In[57]:


df.corr()['mort_acc'].sort_values()


# In[ ]:





# **TASK: Looks like the total_acc feature correlates with the mort_acc , this makes sense! Let's try this fillna() approach. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. To get the result below:**

# In[58]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[ ]:





# **CHALLENGE TASK: Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns. Check out the link below for more info, or review the solutions video/notebook.**
# 
# [Helpful Link](https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe) 

# In[59]:


def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[60]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)


# In[61]:


df.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# **TASK: revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. Go ahead and remove the rows that are missing those values in those columns with dropna().**

# In[62]:


df = df.dropna()


# In[63]:


df.isnull().sum()


# In[ ]:





# ## Categorical Variables and Dummy Variables
# 
# **We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.**
# 
# **TASK: List all the columns that are currently non-numeric. [Helpful Link](https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type)**
# 
# [Another very useful method call](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)

# In[64]:


df.select_dtypes(['object']).columns


# In[ ]:





# ---
# **Let's now go through all the string features to see what we should do with them.**
# 
# ---
# 
# 
# ### term feature
# 
# **TASK: Convert the term feature into either a 36 or 60 integer numeric data type using .apply() or .map().**

# In[65]:


feat_info('term')


# In[66]:


df['term'].value_counts()


# In[67]:


df['term'] = df['term'].apply(lambda term: int(term[:3]))


# In[68]:


df['term']


# ### grade feature
# 
# **TASK: We already know grade is part of sub_grade, so just drop the grade feature.**

# In[69]:


df = df.drop('grade', axis=1)


# In[ ]:





# **TASK: Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe. Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.**

# In[70]:


dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1),dummies],axis=1)


# In[71]:


df.columns


# In[ ]:





# In[ ]:





# In[ ]:





# ### verification_status, application_type,initial_list_status,purpose 
# **TASK: Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

# In[72]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose']],drop_first=True)
df = pd.concat([df.drop(['verification_status', 'application_type','initial_list_status','purpose'], axis=1),dummies],axis=1)


# In[ ]:





# In[ ]:





# ### home_ownership
# **TASK:Review the value_counts for the home_ownership column.**

# In[73]:


df['home_ownership'].value_counts()


# In[ ]:





# **TASK: Convert these to dummy variables, but [replace](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html) NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

# In[74]:


df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'], 'OTHER')


# In[75]:


dummies = pd.get_dummies(df[['home_ownership']],drop_first=True)
df = pd.concat([df.drop(['home_ownership'], axis=1),dummies],axis=1)


# ### address
# **TASK: Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column.**

# In[76]:


df['zip_code'] = df['address'].apply(lambda address: address[-5:])


# In[77]:


df['zip_code'].value_counts()


# **TASK: Now make this zip_code column into dummy variables using pandas. Concatenate the result and drop the original zip_code column along with dropping the address column.**

# In[78]:


dummies = pd.get_dummies(df[['zip_code']],drop_first=True)
df = pd.concat([df.drop(['zip_code'], axis=1),dummies],axis=1)


# In[79]:


df =  df.drop('address', axis=1)


# ### issue_d 
# 
# **TASK: This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.**

# In[80]:


feat_into('issue_d')


# In[81]:


df = df.drop('issue_d', axis=1)


# ### earliest_cr_line
# **TASK: This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.**

# In[82]:


feat_info('earliest_cr_line')


# In[83]:


df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))


# In[84]:


df['earliest_cr_line'].value_counts()


# ## Train Test Split

# **TASK: Import train_test_split from sklearn.**

# In[85]:


from sklearn.model_selection import train_test_split


# **TASK: drop the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.**

# In[86]:


df =  df.drop('loan_status', axis=1)


# In[ ]:





# **TASK: Set X and y variables to the .values of the features and label.**

# In[87]:


X = df.drop('loan_repaid', axis=1).values


# In[88]:


y = df['loan_repaid'].values


# ----
# ----
# 
# # OPTIONAL
# 
# ## Grabbing a Sample for Training Time
# 
# ### OPTIONAL: Use .sample() to grab a sample of the 490k+ entries to save time on training. Highly recommended for lower RAM computers or if you are not using GPU.
# 
# ----
# ----

# In[89]:


# df = df.sample(frac=0.1,random_state=101)
print(len(df))


# **TASK: Perform a train/test split with test_size=0.2 and a random_state of 101.**

# In[90]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[ ]:





# ## Normalizing the Data
# 
# **TASK: Use a MinMaxScaler to normalize the feature data X_train and X_test. Recall we don't want data leakge from the test set so we only fit on the X_train data.**

# In[91]:


from sklearn.preprocessing import MinMaxScaler


# In[92]:


scaler = MinMaxScaler()


# In[93]:


X_train = scaler.fit_transform(X_train)


# In[94]:


X_test= scaler.transform(X_test)


# In[ ]:





# # Creating the Model
# 
# **TASK: Run the cell below to import the necessary Keras functions.**

# In[95]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# **TASK: Build a sequential model to will be trained on the data. You have unlimited options here, but here is what the solution uses: a model that goes 78 --> 39 --> 19--> 1 output neuron. OPTIONAL: Explore adding [Dropout layers](https://keras.io/layers/core/) [1](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) [2](https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab)**

# In[96]:


# CODE HERE
model = Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# In[97]:


X_train.shape


# **TASK: Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting. Optional: add in a batch_size of 256.**

# In[98]:


model.fit(x=X_train,y=y_train,epochs=25,batch_size=256,
         validation_data=(X_test, y_test))


# In[133]:





# **TASK: OPTIONAL: Save your model.**

# In[100]:


from tensorflow.keras.models import load_model


# In[101]:


model.save('myfavoritemodel.h5')


# In[136]:





# # Section 3: Evaluating Model Performance.
# 
# **TASK: Plot out the validation loss versus the training loss.**

# In[103]:


losses = pd.DataFrame(model.history.history)


# In[104]:


losses.plot()


# In[139]:





# **TASK: Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.**

# In[107]:


from sklearn.metrics import  classification_report,confusion_matrix


# In[108]:


predictions = model.predict_classes(X_test)


# In[110]:


print(classification_report(y_test, predictions))


# In[143]:





# In[144]:





# **TASK: Given the customer below, would you offer this person a loan?**

# In[111]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[114]:


new_customer = scaler.transform(new_customer.values.reshape(1,78)) 


# In[116]:


model.predict_classes(new_customer)


# **TASK: Now check, did this person actually end up paying back their loan?**

# In[119]:


df.iloc[random_ind]['loan_repaid']


# In[149]:





# # GREAT JOB!
