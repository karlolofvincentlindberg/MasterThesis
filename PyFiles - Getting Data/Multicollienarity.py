import pandas as pd
import numpy as np
import seaborn as sns
import math as math
import matplotlib.pyplot as plt
import datetime as dt
import scipy.linalg as la
import scipy.optimize as optimize
from patsy import dmatrices

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Linear
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import linear_model

from datetime import date
from pandas import ExcelWriter

# Dividing the data
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"
tickers = ["AMZN", "AAPL", "META", "GOOGL", "MSFT", "NFLX"]
frequencies = ["Daily", "Weekly", "Monthly"]

# Listing all variables
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

# Sentiment
sentiment_list = [tpsc, tnsc, npsc, nnsc, pcr, ivm]
attention_list = [tvl, tpc, npc, svi]

list_all = []
for i in range(len(attention_list)):
    list_all.append(attention_list[i])
for i in range(len(sentiment_list)):
    list_all.append(sentiment_list[i])

mindate = dt.datetime(2015, 2, 1)
mindate = mindate.strftime('%Y-%m-%d')


def read_xl(r, d):
    ticker = tickers[r]
    frequency = frequencies[d]
    filename = directory+"/RatioDATA/"+ticker+".xlsx"
    data = pd.read_excel(filename, index_col=0, engine=None,
                         sheet_name=ticker+" "+frequency)
    data = data[data.index >= mindate]
    return data


def read_cs(r, d):
    ticker = tickers[r]
    frequency = frequencies[d]
    file = directory+"/18 CSVs - Final/"+ticker+" - "+frequency+".csv"
    data = pd.read_csv(file, index_col=0, engine=None)
    return data


def vif(data, list):
    y = data['Price']
    x = data[list]
    x.insert(0, 'Intercept', 1.0)
    # y, x = dmatrices('Price ~ ATTN+SENT', data=data, return_type='dataframe')
    # print(x)
    vif_df = pd.DataFrame()
    vif_df['variable'] = x.columns
    vif_df['VIF'] = [variance_inflation_factor(
        x.values, i) for i in range(len(x.columns))]
    return vif_df


for i in range(len(tickers)):
    for j in range(len(frequencies)):
        print(tickers[i]+" "+frequencies[j])
        data = read_xl(i, j)
        vi = vif(data, ['ATTN', 'SENT'])
        print(vi)

for i in range(len(tickers)):
    for j in range(len(frequencies)):
        print(tickers[i]+" "+frequencies[j])
        data = read_cs(i, j)
        vi_all = vif(data, list_all)
        vi_at = vif(data, attention_list)
        vi_se = vif(data, sentiment_list)
        print(vi_all)
        print(vi_at)
        print(vi_se)
