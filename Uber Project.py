#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import os


# ## Loading and Preparation

#  Loading Data

# In[3]:


files=os.listdir(r'C:\Users\ash.ASHISH\Projects\Project 1..Uber New York Trip\uber-pickups-in-new-york-city')[-7:]
files


# In[4]:


files.remove('uber-raw-data-janjune-15.csv')


# In[5]:


files


# In[6]:


path=r'C:\Users\ash.ASHISH\Projects\Project 1..Uber New York Trip\uber-pickups-in-new-york-city'

final=pd.DataFrame()
for file in files:
    df=pd.read_csv(path+"/"+file)
    final=pd.concat([df,final])


# In[7]:


final.shape


# ### Data Preparation¶

# In[8]:


df=final.copy()


# In[9]:


df.head()


# In[10]:


df.dtypes


# In[11]:


df['Date/Time']=pd.to_datetime(df['Date/Time'],format='%m/%d/%Y %H:%M:%S')


# In[12]:


df.dtypes


# In[13]:


df.head()


# In[14]:


df['Weekday']=df['Date/Time'].dt.day_name()


# In[15]:


df['Day']=df['Date/Time'].dt.day


# In[16]:


df['Minute']=df['Date/Time'].dt.minute
df['Month']=df['Date/Time'].dt.month
df['Hour']=df['Date/Time'].dt.hour


# In[17]:


df.head()


# In[18]:


df.dtypes


# ### Analysis of journey by Week-days

# In[19]:


import plotly.express as px


# In[20]:


px.bar(x=df['Weekday'].value_counts().index,
      y=df['Weekday'].value_counts()
      )


# **seems to have highest sales on Thrusday**

# ### Analysis by Hour

# In[21]:


plt.hist(df['Hour'])


# In[22]:


df['Month'].unique()


# In[23]:


plt.figure(figsize=(40,20))
for i, Month in enumerate(df['Month'].unique()):
    plt.subplot(3,2,i+1)
    df[df['Month']==Month]['Hour'].hist()


# ### Analysis of Rush of each hour in each month

# In[24]:


for i in df['Month'].unique():
    plt.figure(figsize=(5,3))
    df[df['Month']==i]['Hour'].hist()


# ###  analysis of which month has max rides

# In[27]:


import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[28]:


trace1 = go.Bar(
        x = df.groupby('Month')['Hour'].sum().index,
        y= df.groupby('Month')['Hour'].sum(),
        name='priority')
iplot([trace1])


# ### Analysis of Journey of Each Day

# In[104]:


plt.figure(figsize=(10,6))
plt.hist(df['Day'],bins=30,rwidth=.8,range=(0.5,30.5))
plt.xlabel('date of the month')
plt.ylabel('Total Journeys')
plt.title('Journeys by Month Day')


# In[ ]:





# ### Analysis of Total rides month wise

# In[102]:


plt.figure(figsize=(10,8))
for i,Month in enumerate(df['Month'].unique(),1):
    plt.subplot(3,2,i)
    df_out=df[df['Month']==Month]
    plt.hist(df_out['Day'])
    plt.xlabel('Days in Month{}'.format(Month))
    plt.ylabel('Total_rides')


# In[ ]:





# ### Analysing Rush in hour

# In[101]:


plt.figure(figsize=(8,6))
sns.set_style(style='whitegrid')
sns.pointplot(x="Hour",y="Lat",data=df)


# **adding hue params**

# In[100]:


plt.figure(figsize=(8,6))
ax=sns.pointplot(x="Hour",y="Lat", hue="Weekday",data=df)
ax.set_title('Hoursoffday vs Latiitude of passenger')


# 

# ### to analyse which base number get popular by month name

# In[33]:


df.head()


# In[34]:


df['Base'].head()


# In[99]:


df.groupby(['Base','Month'])['Date/Time'].count()


# In[97]:


base=df.groupby(['Base','Month'])['Date/Time'].count().reset_index()
base.head()                


# #### to analyse which base number gets popular by month name

# In[96]:


plt.figure(figsize=(8,6))
sns.lineplot(x='Month',y='Date/Time',hue='Base',data=base)


# ### 2 Cross Analysis
# #### Through our exploration we are going to visualize:
# #### 1.Heatmap by Hour and Weekday.
# #### 2.Heatmap by Hour and Day.
# #### 3.Heatmap by Month and Day.
# #### 4.Heatmap by Month and Weekday.

# ## Heatmap by Hour and Weekday.

# In[38]:


def count_rows(rows):
    return len(rows)


# In[39]:


by_cross = df.groupby(['Weekday','Hour']).apply(count_rows)
by_cross


# In[40]:


pivot = by_cross.unstack()
pivot


# ### creating heatmap for visualize

# In[95]:


plt.figure(figsize=(8,6))
sns.heatmap(pivot)


# In[89]:


def heatmap(col1,col2):
    by_cross=df.groupby([col1,col2]).apply(lambda x:len(x))
    pivot=by_cross.unstack()
    plt.figure(figsize=(8,6))
    return sns.heatmap(pivot,annot=False)


# In[44]:


heatmap('Day','Hour')


# In[45]:


heatmap('Day','Month')


# ### Analysing the results
# #### We observe that the number of trips increases each month, we can say that from April to September 2014, Uber was in a continuous improvement process.

# In[46]:


df[df['Month']==4]


# In[47]:


heatmap('Weekday','Month')


# #### Analysis of Location data points¶

# In[88]:


plt.figure(figsize=(8,6))

plt.plot(df['Lon'], df['Lat'],'r+', ms=0.5)
plt.xlim(-74.2, -73.7)
plt.ylim(40.6,41)


# ##### We can see a number of hot spots here. Midtown Manhattan is clearly a huge bright spot.
# ##### & these are made from Midtown to Lower Manhattan.
# ##### Followed by Upper Manhattan and the Heights of Brooklyn.

# In[ ]:





# ### perform Spatial Analysis using heatmap to get a clear cut of Rush on Sunday(Weekend)

# In[49]:


df_out=df[df['Weekday']=='Sunday']
df_out.head()


# In[50]:


rush=df_out.groupby(['Lat','Lon'])['Weekday'].count().reset_index()
rush.head()


# In[51]:


#!pip install folium


# In[52]:


from folium.plugins import HeatMap


# In[53]:


import folium


# In[54]:


basemap=folium.Map()


# In[55]:


HeatMap(rush,zoom=20,radius=15).add_to(basemap)
basemap


# In[ ]:





# ### Lets create a function for a specific day

# In[90]:


def plot(df,Day):
    df_out = df[df['Weekday']==Day]
    df_out.groupby(['Lat','Lon'])['Weekday'].count().reset_index()
    HeatMap(df_out.groupby(['Lat','Lon'])['Weekday'].count().reset_index(),zoom=20,radius=15).add_to(basemap)
    return basemap


# In[91]:


plot(df,'Sunday')


# In[ ]:





# ## Analysis of Jan-June uber_15

# In[58]:


uber_15=pd.read_csv(r'C:\Users\ash.ASHISH\Projects\Project 1..Uber New York Trip\uber-pickups-in-new-york-city/uber-raw-data-janjune-15.csv')
uber_15.head()


# In[59]:


uber_15.dtypes


# In[60]:


#Checking the minimum date in the uber_15
uber_15['Pickup_date'].min()


# In[61]:


#Checking the maximum date in the uber_15
uber_15['Pickup_date'].max()


# In[62]:


uber_15['Pickup_date']=pd.to_datetime(uber_15['Pickup_date'],format='%Y-%m-%d %H:%M:%S')


# In[63]:


uber_15.dtypes


# In[64]:


uber_15['Weekday']=uber_15['Pickup_date'].dt.day_name()
uber_15['Day']=uber_15['Pickup_date'].dt.day
uber_15['Month']=uber_15['Pickup_date'].dt.month
uber_15['Minute']=uber_15['Pickup_date'].dt.minute
uber_15['Hour']=uber_15['Pickup_date'].dt.hour


# In[65]:


uber_15.head()


# ### Uber pickups by the month in NYC

# In[85]:


plt.figure(figsize=(8,6))
px.bar(x=uber_15['Month'].value_counts().index,
      y=uber_15['Month'].value_counts().values)


# #### We can see that the number of Uber pickup has been steadily increasing throughout the first half of 2015 in NYC

# In[ ]:





# #### Analysing Rush in New york City

# In[84]:


plt.figure(figsize=(8,6))
sns.countplot(uber_15['Hour'])


# ####  Interestingly, after the morning rush, the number of Uber pickups doesn't dip much throughout the rest of the morning and early afternoon. There is significantly more demand in the evening than the daytime. Let's investigate to see if there's a difference in hourly pattern for different days of the week.

# In[ ]:





# ### Analysing In-Depth Analysis of Rush in New york City Day & hour wise

# In[68]:


summary=uber_15.groupby(['Weekday','Hour'])['Pickup_date'].count().reset_index()
summary.head()


# In[69]:


summary.columns=['Weekday','Hours','Counts']


# In[70]:


summary.head()


# In[92]:


plt.figure(figsize=(8,6))
sns.pointplot(x='Hours',y='Counts',hue='Weekday',data=summary)


# In[ ]:





# In[ ]:





# ##### Loading Uber-Jan-Feb-FOIL.csv

# In[72]:


uber_foil=pd.read_csv(r'C:\Users\ash.ASHISH\Projects\Project 1..Uber New York Trip\uber-pickups-in-new-york-city/Uber-Jan-Feb-FOIL.csv')


# In[73]:


uber_foil.head()


# In[74]:


uber_foil['dispatching_base_number'].unique()


# In[75]:


sns.boxplot(x = 'dispatching_base_number', y = 'active_vehicles', data = uber_foil)


# #### seems to have more number of Active Vehicles in B02764

# In[ ]:





# In[76]:


sns.boxplot(x = 'dispatching_base_number', y = 'trips', data = uber_foil)


# #### seems to have more number of trips in B02764

# In[ ]:





# In[77]:


# Finding the ratio of trips/active_vehicles
uber_foil['trips/vehicle'] = uber_foil['trips']/uber_foil['active_vehicles']


# In[78]:


uber_foil.head()


# In[79]:


uber_foil.set_index('date').head()


# ##### how Average trips/vehicle inc/decreases with dates with each of base umber

# In[94]:


plt.figure(figsize=(10,6))
uber_foil.set_index('date').groupby(['dispatching_base_number'])['trips/vehicle'].plot()
plt.ylabel('Average trips/vehicle')
plt.title('Demand vs Supply chart (Date-wise)')
plt.legend()


# In[ ]:




