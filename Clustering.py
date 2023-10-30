#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#Reading excel sheet for the clustering

excel_sheet = pd.read_excel(r'C:\Users\nicol\Downloads\data.xlsx')


# In[3]:


#Display of the first 5 values

excel_sheet.head()


# In[4]:


#Dimensions of data table

excel_sheet.shape


# In[2]:


#Display of missing values in NaN

excel_sheet = pd.read_excel(r'C:\Users\nicol\Downloads\data.xlsx',na_values = excel_sheet.loc[1,"Net Debt/Equity"])


# In[3]:


excel_sheet.dtypes


# In[7]:


#Display of correlations

sns.heatmap(excel_sheet.corr());


# In[8]:


#Decription of the differents columns (ratios)

excel_sheet.describe()


# In[9]:


#Number of missing values per column

excel_sheet.isna().sum()


# In[10]:


#First remarks

excel_sheet.sector.unique()


# In[11]:


excel_sheet['sub sector'].unique()


# In[12]:


len(excel_sheet['sub sector'].unique())


# In[13]:


#There are 12 sub-sectors of activity represented 
#The objective is to use an unsupervised learning technique (KMeans) to find the appropriate number of clusters K
#We use the Elbow method to determine K
#The elbow method is used by running several k-means, increment k with each iteration, and record the SSE 
#𝑆𝑆𝐸=𝑆𝑢𝑚 𝑂𝑓 𝐸𝑢𝑐𝑙𝑖𝑑𝑒𝑎𝑛 𝑆𝑞𝑢𝑎𝑟𝑒𝑑 𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒𝑠 𝑜𝑓 𝑒𝑎𝑐ℎ 𝑝𝑜𝑖𝑛𝑡 𝑡𝑜 𝑖𝑡𝑠 𝑐𝑙𝑜𝑠𝑒𝑠𝑡 𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑 (decreasing function of K)

from sklearn.cluster import KMeans

var = []
for i in range(2,12):
    group = KMeans(n_clusters=i).fit(excel_sheet.select_dtypes("number").dropna())
    var.append(group.inertia_)
    

plt.plot(range(2,12),var)


# In[14]:


get_ipython().system('pip install kneed')


# In[15]:


import kneed


# In[16]:


# get elbow programmatically
from kneed import KneeLocator 
kl = KneeLocator(
range(2,12), var, curve="convex", direction="decreasing")
elbow=kl.elbow
print('Elbow = {}'.format(elbow))


# In[17]:


# We apply KMeans for the Elbow's value  ( in this case = 4)
kmeans = KMeans(n_clusters=elbow)
km = kmeans.fit(excel_sheet.select_dtypes("number").dropna())

dropna = excel_sheet.select_dtypes("number").dropna()
dropna["cluster"] = km.labels_


# In[18]:


sns.scatterplot(x="Return on Assets", y="Total Debt", hue="cluster", data=dropna);


# In[19]:


dropna["cluster"].value_counts()


# In[20]:


dropna[dropna["cluster"]==1]


# In[21]:


var2 = []
for i in range(2,40):
    group2 = KMeans(n_clusters=i).fit(excel_sheet[['Return on Assets', 'Return on Invested Capital', 'EBITDA Margin']])
    var2.append(group2.inertia_)
    

plt.plot(range(2,40),var2)


# In[22]:


# get elbow programmatically
from kneed import KneeLocator 
kl2 = KneeLocator(
range(2,40), var2, curve="convex", direction="decreasing")
elbow2=kl2.elbow
print('Elbow = {}'.format(elbow2))


# In[23]:


# We apply KMeans for the Elbow's value  ( in this case = 4)
kmeans = KMeans(n_clusters=elbow2)
km2 = kmeans.fit(excel_sheet[['Return on Assets', 'Return on Invested Capital', 'EBITDA Margin']])
excel_sheet["cluster"]=km2.labels_


# In[24]:


sns.scatterplot(x="Return on Assets", y="Return on Invested Capital", hue="cluster", data=excel_sheet);


# In[33]:


excel_sheet2= excel_sheet.drop(41,axis=0)
excel_sheet2.shape
#sns.scatterplot(x="Return on Invested Capital", y="EBITDA Margin", hue="cluster", data=excel_sheet);


# In[34]:


sns.scatterplot(x='Return on Invested Capital', y='EBITDA Margin', hue='cluster', data=excel_sheet2);


# In[35]:


excel_sheet2['cluster'].value_counts()


# In[36]:


km3 = kmeans.fit(excel_sheet[['Return on Assets', 'Return on Invested Capital', 'EBITDA Margin', 'Net Income Margin', 'EV/EBITDA']])
excel_sheet["cluster"]=km3.labels_


# In[37]:


excel_sheet2= excel_sheet.drop(41,axis=0)


# In[38]:


sns.scatterplot(x='Return on Invested Capital', y='EBITDA Margin', hue='cluster', data=excel_sheet2);


# In[39]:


sns.scatterplot(x='Return on Invested Capital', y='Net Income Margin', hue='cluster', data=excel_sheet2);


# In[41]:


var2 = []
for i in range(2,40):
    group2 = KMeans(n_clusters=i).fit(excel_sheet[['Return on Assets', 'Return on Invested Capital', 'EBITDA Margin', 'Net Income Margin', 'EV/EBITDA']])
    var2.append(group2.inertia_)
    

plt.plot(range(2,40),var2)


# In[42]:


# get elbow programmatically
from kneed import KneeLocator 
kl2 = KneeLocator(
range(2,40), var2, curve="convex", direction="decreasing")
elbow2=kl2.elbow
print('Elbow = {}'.format(elbow2))


# In[43]:


km3 = KMeans(n_clusters=elbow2).fit(excel_sheet[['Return on Assets', 'Return on Invested Capital', 'EBITDA Margin', 'Net Income Margin', 'EV/EBITDA']])
excel_sheet["cluster"]=km3.labels_


# In[45]:


excel_sheet2= excel_sheet.drop(41,axis=0)
sns.scatterplot(x='Return on Invested Capital', y='EBITDA Margin', hue='cluster', data=excel_sheet2);


# In[4]:


#For standardizing features, We'll use the StandardScaler module
from sklearn.preprocessing import StandardScaler


# In[5]:


scaler=StandardScaler()
excel_sheet_std = scaler.fit_transform(excel_sheet.select_dtypes('number').dropna())


# In[6]:


len(excel_sheet_std)


# In[7]:


from sklearn.decomposition import PCA


# In[8]:


pca = PCA()
pca.fit(excel_sheet_std)


# In[9]:


pca.explained_variance_ratio_


# In[10]:


excel_sheet_std[0]


# In[11]:


excel_sheet.dtypes


# In[12]:


excel_sheet = pd.read_excel(r'C:\Users\nicol\Downloads\data.xlsx')
excel_sheet.dtypes


# In[13]:


excel_sheet = pd.read_excel(r'C:\Users\nicol\Downloads\data.xlsx',na_values = excel_sheet.loc[1,"Net Debt/Equity"])


# In[14]:


excel_sheet.dtypes


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scaler=StandardScaler()
excel_sheet_std = scaler.fit_transform(excel_sheet.select_dtypes('number').dropna())


# In[17]:


excel_sheet_std[0:3]


# In[18]:


from sklearn.decomposition import PCA


# In[19]:


pca = PCA()
pca.fit(excel_sheet_std)


# In[20]:


pca.explained_variance_ratio_


# In[21]:


plt.plot(range(1,11),pca.explained_variance_ratio_.cumsum())
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')


# In[22]:


pca = PCA(n_components = 4)


# In[23]:


pca.fit(excel_sheet_std)


# In[24]:


result_pca = pca.transform(excel_sheet_std)


# In[25]:


from sklearn.cluster import KMeans

var_pca = []
for i in range(2,20):
    kmeans_pca = KMeans(n_clusters=i)
    kmeans_pca.fit(result_pca)
    var_pca.append(kmeans_pca.inertia_)
    

plt.plot(range(2,20),var_pca,marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squares')
plt.title('KMeans with PCA Clustering')


# In[26]:


# get elbow programmatically
from kneed import KneeLocator 
kl = KneeLocator(
range(2,20), var_pca, curve="convex", direction="decreasing")
elbow=kl.elbow
print('Elbow = {}'.format(elbow))


# In[27]:


kmeans_pca = KMeans(n_clusters=6)
kmeans_pca.fit(result_pca)


# In[38]:


excel_sheet_pca = pd.concat([excel_sheet.dropna().reset_index(drop=True), pd.DataFrame(result_pca)],axis=1)
excel_sheet_pca.columns.values[-4:] = ['Component 1', 'Component 2', 'Component 3', 'Component 4']
excel_sheet_pca['PCA cluster'] = kmeans_pca.labels_


# In[39]:


excel_sheet_pca.head()


# In[41]:


sns.scatterplot('Component 1','Component 2', hue='PCA cluster', data = excel_sheet_pca)


# In[ ]:




