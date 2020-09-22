#!/usr/bin/env python
# coding: utf-8

# In[5]:


from gs_quant.session import GsSession, Environment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

GsSession.use(client_id="id", client_secret="secret")


# In[6]:


from gs_quant.data import Dataset
import datetime

def get_datasets(datasets):
    ds_dict = {}
    for dataset in datasets:
        try:
            df = Dataset(dataset).get_data(datetime.date(2020, 6, 24), datetime.datetime.today().date())
            
            keys = [x for x in ['countryId', 'subdivisionId'] if x in df.columns] + ['date']
            val_map = {'newConfirmed': 'totalConfirmed', 'newFatalities': 'totalFatalities'}
            vals = [x for x in list(val_map.keys()) if x in df.columns]

            df_t = df.groupby(keys).sum().groupby(level=0).cumsum().reset_index()[keys + vals].rename(columns=val_map)
            ds_dict[dataset] = df.reset_index().merge(df_t, on=keys, suffixes=('', '_y')).set_index('date')

        except Exception as err:
            print(f'Failed to obtain {dataset} with {getattr(err,"message",repr(err))}')
    return ds_dict


# In[8]:


country_datasets = [
    'COVID19_COUNTRY_DAILY_WHO',
]
df = get_datasets(country_datasets)


# In[88]:


import matplotlib.pyplot as plt
import pandas as pd


def plottovix(countryId):
    dataset='COVID19_COUNTRY_DAILY_WHO'     

    frame = Dataset(dataset).get_data(start_date=datetime.date(2020, 1, 1), countryId=countryId)

    vix = pd.read_csv('./vix.csv', index_col='Date')

    vix.index = pd.to_datetime(vix.index)

    fig,ax = plt.subplots()

    fig.set_size_inches(18.5, 10.5)
    # make a plot
    ax.plot(vix['Adj Close'], color="red", marker="o")
    ax.set_xlabel("Date",fontsize=14)
    ax.set_ylabel("VIX",color="red",fontsize=14)
    ax2=ax.twinx()
    ax2.plot(frame['newConfirmed'],color="blue",marker="o")
    ax2.set_ylabel("New Cases",color="blue",fontsize=14)
    plt.show()


# In[39]:


import numpy as np

ind = frame[22:].index.union(vix[13:].index)

frame['% change'] = frame['totalConfirmed'].shift(-8)/frame['totalConfirmed'].shift(-7) -1
fig,ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
# make a plot
ax.plot(vix['Adj Close'], color="red", marker="o")
ax.set_xlabel("Date",fontsize=14)
ax.set_ylabel("VIX",color="red",fontsize=14)

ax2=ax.twinx()
ax2.plot(frame['totalConfirmed'].shift(7)/frame['totalConfirmed'].shift(8) -1,color="blue",marker="o")
ax2.set_ylabel("Daily % Change",color="blue",fontsize=14)
plt.show()
print(np.corrcoef(vix.reindex(ind)['Adj Close'].fillna(0),frame.reindex(ind)['% change'].fillna(0)))


# In[40]:


vix['% change'] = vix['Adj Close']/vix['Adj Close'].shift(1) -1
fig,ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
# make a plot
ax.plot(vix['Adj Close']/vix['Adj Close'].shift(1) -1 , color="red", marker="o")
ax.set_xlabel("Date",fontsize=14)
ax.set_ylabel("VIX % change",color="red",fontsize=14)

ax2=ax.twinx()
ax2.plot(frame['% change'],color="blue",marker="o")
ax2.set_ylabel("Daily % Change",color="blue",fontsize=14)
plt.show()
#print(np.corrcoef(us['total_cases']/us['total_cases'].shift(1)-1,(vix['Adj Close']/vix['Adj Close'].shift(1) -1)[65:]))

print(np.corrcoef(vix.reindex(ind)['% change'].fillna(0),frame.reindex(ind)['% change'].fillna(0)))


# In[42]:


ind20 =  frame[frame['% change'] > .25].index 

vixcop = vix.copy()

vix.reindex(ind)['% change'].mean()


# In[49]:


coverage = Dataset(dataset).get_coverage()


# In[50]:


print(coverage)


# In[77]:


print(coverage['countryId'])

print(Dataset(dataset).get_data(start_date=datetime.date(2020, 1, 1), countryId='NI'))


# In[90]:


coverage = Dataset(dataset).get_coverage()

for country in coverage['countryId']:
    try:
        countryData = 
            Dataset(dataset).get_data
            (start_date=datetime.date(2020, 1, 1), 
             countryId=country)
        maxDate = 
            countryData['newConfirmed'].idxmax()
        if(maxDate.dayofyear>60 and 
           maxDate.dayofyear<90):
            plottovix(country) 
    except KeyError:
        


        
    


# In[99]:


plottovix('CN')


# In[98]:


vix = pd.read_csv('./vix.csv', index_col='Date')

vix.index = pd.to_datetime(vix.index)

fig,ax = plt.subplots()

fig.set_size_inches(18.5, 10.5)

ax.plot(vix['Adj Close'], color="red", marker="o")
ax.set_xlabel("Date",fontsize=14)
ax.set_ylabel("VIX",color="red",fontsize=14)
    
plt.show()


# In[ ]:




