import pandas as pd
import numpy as np
import seaborn as sns
import math as math
import matplotlib.pyplot as plt
import datetime as dt
import scipy.linalg as la
import scipy.optimize as optimize

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

# Linear
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import linear_model

from datetime import date
from pandas import ExcelWriter

# Dividing the data
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"
tickers = ["AMZN", "AAPL", "META", "GOOGL", "MSFT", "NFLX"]

# Function to read in our raw data


def read_data(r, freq):
    file = directory+"/18 CSVs - Final/"+tickers[r]+" - "+freq+".csv"
    data = pd.read_csv(file, index_col=0, engine=None)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data = data.set_index('Date')
    data['TR'] = data['TV']/data['SHOUT']
    return data


# Our frequencies
frequencies = ["Daily", "Weekly", "Monthly"]

# Functions to create lists from columns of dataset


def create_lists(cols):
    list1 = []
    list2 = []

    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            label = f'{ticker} {frequency}'
            list1.append(label)
    for i in range(len(cols)):
        label1 = f'{cols[i]} - ADF'
        label2 = f'{cols[i]} - KPSS'
        list2.append(label1)
        list2.append(label2)
    return list1, list2


# Our variables
tvl = 'TV'
tpc = 'TC'
tpsc = 'TPC'
tnsc = 'TNC'
npc = 'NC'
npsc = 'NPC'
nnsc = 'NNC'
si = 'SI'
sir = 'SIR'
ivm = 'VMP'
pcr = 'PCR'
ti = 'SVI - T'
sna = 'SVI - S'
svi = 'SVI'
sho = 'SHOUT'

# Sentiment & attention lists
sentiment_list = [tpsc, tnsc, npsc, nnsc, pcr, ivm]
attention_list = [tvl, tpc, npc, svi]


mindate = dt.datetime(2015, 2, 1)
mindate = mindate.strftime('%Y-%m-%d')
maxdate = dt.datetime(2022, 1, 1)
maxdate = maxdate.strftime('%Y-%m-%d')

# Creating a list of all the raw variables we want to check
list_mix = ['Price']
for i in range(len(sentiment_list)):
    list_mix.append(sentiment_list[i])
for i in range(len(attention_list)):
    list_mix.append(attention_list[i])
list_mix.append('TR')

# Checking the stationarity for our raw varibles


def collinearity_matrix(cols, name):
    labels, column_list = create_lists(cols)
    dfmat = pd.DataFrame(data=None, index=labels, columns=column_list)
    for k in range(len(cols)):
        for j in range(len(frequencies)):
            frequency = frequencies[j]
            for i in range(len(tickers)):
                data = read_data(i, frequency)
                data = data.loc[:, list_mix]
                data = data[data.index >= mindate]
                data = data[data.index <= maxdate]
                data = data.diff()
                result = adfuller(data.iloc[1:, k])

                r = j*6+i
                c = 2*k
                if result[1] > 0.05:
                    dfmat.iloc[r, c] = "No"
                else:
                    dfmat.iloc[r, c] = "Yes"
    for k in range(len(cols)):
        for j in range(len(frequencies)):
            frequency = frequencies[j]
            for i in range(len(tickers)):
                data = read_data(i, frequency)
                data = data.loc[:, list_mix]
                data = data[data.index >= mindate]
                data = data[data.index <= maxdate]
                data = data.diff()
                kpss_stat, p_value, lags, critical_values = kpss(
                    data.iloc[1:, k])
                r = j*6+i
                c = 2*k+1
                if p_value < 0.05:
                    dfmat.iloc[r, c] = "No"
                else:
                    dfmat.iloc[r, c] = "Yes"
    dfmat.to_excel(f"{directory}/StationaritY/{name}.xlsx")
    return dfmat


df_mat = collinearity_matrix(list_mix, "Stationarity")

df_trans = df_mat.transpose()

df_daily = df_trans.iloc[:, :6]
df_daily.columns = tickers
with open(directory+'/LatexTables/Stationarities/raw_sta_d.tex', 'w') as tf:
    tf.write(df_daily.to_latex())

df_weekly = df_trans.iloc[:, 6:12]
df_weekly.columns = tickers
with open(directory+'/LatexTables/Stationarities/raw_sta_w.tex', 'w') as tf:
    tf.write(df_weekly.to_latex())

df_monthly = df_trans.iloc[:, 12:]
df_monthly.columns = tickers
with open(directory+'/LatexTables/Stationarities/raw_sta_m.tex', 'w') as tf:
    tf.write(df_monthly.to_latex())
