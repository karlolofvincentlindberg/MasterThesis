import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import datetime as dt
from datetime import date

# Our directory
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"

# Reading the data
dataname = directory+"/Data/IVAll.csv"
data = pd.read_csv(dataname)

tickers = ["AMZN", "AAPL", "META", "GOOGL", "MSFT", "NFLX"]
data_tickers = data['Ticker Symbol'].unique()

column_list = ['Days to Expiration',
               'Ticker Symbol', 'The Date of this Option Price']

data['The Date of this Option Price'] = pd.to_datetime(
    data['The Date of this Option Price'])
data['The Date of this Option Price'] = data['The Date of this Option Price'].dt.strftime(
    '%Y-%m-%d')
data = data.sort_values(by=['Ticker Symbol', 'The Date of this Option Price'])

data.drop(data[data['Days to Expiration'] > 30].index, inplace=True)
data.drop(data[data['Days to Expiration'] < 30].index, inplace=True)

vol = data.groupby(column_list).count()
sumvol = data.groupby(column_list)['Implied Volatility of the Option'].sum()

midvol = pd.DataFrame(index=vol.index)
midvol['sumvol'] = sumvol
# midvol['count'] = vol['Security ID']
midvol['mid'] = midvol['sumvol']/2

midvol = midvol.reset_index()
midvol = midvol.set_index('Ticker Symbol')

midvolcomp = pd.DataFrame(
    index=midvol['The Date of this Option Price'].unique())
dates = midvol['The Date of this Option Price'].unique()

x = midvol.loc['META', 'The Date of this Option Price':'mid']
y = len(dates)-x['The Date of this Option Price'].count()

for i in range(len(data_tickers)):
    name = tickers[i]
    midvolcomp[name] = np.nan
    length = len(dates)
    for j in range(length):
        dat = midvol[midvol['The Date of this Option Price'] == dates[j]]
        if i != 2:
            midvolcomp.iloc[j, i] = dat.loc[name, 'mid']
        elif j >= y:
            midvolcomp.iloc[j, i] = dat.loc[name, 'mid']
        else:
            x = 0

midvolcomp.to_csv(directory+"/Data/MidVol.csv")
