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
    for i in range(1):
        label1 = f'{cols} - ADF'
        label2 = f'{cols} - KPSS'
        list2.append(label1)
        list2.append(label2)
    return list1, list2


# Checking stationarity for our computed ratios - residualized data
# Reading in dataset to get columns and labels for results matrix
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/SavedData/ResData/"+ticker+" "+frequency+".xlsx"
data = pd.read_excel(filename, index_col=0, engine=None)
colsss = data.columns[-3]
labels, column_list = create_lists(colsss)

# Creating dataframe to hold our results
dfmat_res = pd.DataFrame(data=None, index=labels, columns=column_list)

# Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe

for j in range(len(frequencies)):
    frequency = frequencies[j]
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/ResData/"+ticker+" "+frequency+".xlsx"
        data = pd.read_excel(filename, index_col=0, engine=None)
        result = adfuller(data.iloc[1:, -3])
        r = j*6+i
        c = 0
        if result[1] > 0.05:
            dfmat_res.iloc[r, c] = "No"
        else:
            dfmat_res.iloc[r, c] = "Yes"

# KPSS test - adding results to matrix and dataframe
for j in range(len(frequencies)):
    frequency = frequencies[j]
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/ResData/"+ticker+" "+frequency+".xlsx"
        data = pd.read_excel(filename, index_col=0, engine=None)
        kpss_stat, p_value, lags, critical_values = kpss(data.iloc[1:, -3])
        r = j*6+i
        c = 1
        if p_value < 0.05:
            dfmat_res.iloc[r, c] = "No"
        else:
            dfmat_res.iloc[r, c] = "Yes"
dfmat_res.to_excel(f"{directory}/Stationarity/ResResultsStationarity.xlsx")

# eXPORTING as table
cols = [dfmat_res.columns[0], dfmat_res.columns[1]]
df_index = [" "]+cols+[" "]+cols+[" "]+cols
df_res = pd.DataFrame(index=df_index, columns=tickers)

df_res.iloc[0, :] = ["Daily"]*6
df_res.iloc[1:3, :] = dfmat_res.transpose().iloc[:, :6]
df_res.iloc[3, :] = ["Weekly"]*6
df_res.iloc[4:6, :] = dfmat_res.transpose().iloc[:, 6:12]
df_res.iloc[6, :] = ["Monthly"]*6
df_res.iloc[7:9, :] = dfmat_res.transpose().iloc[:, 12:]
with open(directory+'/LatexTables/Stationarities/Stationarity_res.tex', 'w') as tf:
    tf.write(df_res.to_latex())
