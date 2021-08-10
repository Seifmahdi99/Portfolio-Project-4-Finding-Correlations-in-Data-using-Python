#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)  #Adjusts the configuration of the plots we will create



#Read in the Data

df= pd.read_csv(r'C:\Users\Amr_M\Downloads\Portfolio Projects\Project 4\movies1.csv')


# In[2]:


# Let's look at the Data

df.head()


# In[3]:


# let's see if there is any missing data 

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[4]:


# Data Types for Columns

df.dtypes


# In[5]:


# Change Datatype for Columns
df['budget'] = df['budget'].astype("Int64")

df['gross'] = df['gross'].astype('Int64')


# In[6]:


df


# In[82]:


# Create correct year column

df['yearcorrect'] = df['released'].astype(str).str.split(',').str[1].str.split().str[1]


df['yearcorrect']

df


# In[100]:


df = df.sort_values(by=['gross'], inplace = False, ascending = False)


# In[86]:


pd.set_option('display.max_rows', None)


# In[88]:


# Drop Dupes

 df.drop_duplicates()


# In[89]:


df


# In[92]:


# Budget high correlation
# Company high correlation 
df.dtypes


# In[109]:


# Scatter Plot with Budget vs Gross
df['budget'] = df['budget'].astype("float")

df['gross'] = df['gross'].astype('float')

plt.scatter(x=df['budget'],y=df['gross'])

plt.title('Budget vs Gross Earnings')

plt.xlabel('Gross Earnings')

plt.ylabel('Budget for Film')

plt.show()


# In[117]:


df.head()


# In[121]:


# Plot the Budget vs Gross using seaborn

sns.regplot(x='budget',y='gross',data=df,scatter_kws={"color" : "red"},line_kws={"color": "blue"})


# In[130]:


# Let's start looking at correlation

df.corr(method = 'pearson') #Pearson , Kendall , Spearman


# In[137]:


# Hypothesis of High Correalation between Budget and Gross was right

correlation_matrix = df.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title("Correlation Matrix to show the correlation between all the numeric variables")

plt.xlabel("Movie Features")

plt.ylabel("Movie Features")

plt.show()


# In[138]:


# looking at Company

df.head()


# In[147]:


df.dtypes


# In[156]:


corr_pairs=correlation_matrix.unstack()
corr_pairs.sort_values()
corr_pairs


# In[158]:


# Finding Highly correlated values
highcorr = corr_pairs[(corr_pairs) > 0.5]

highcorr


# In[159]:


# Votes and Budget has the highest correlation to gross
# Company is not the highest

