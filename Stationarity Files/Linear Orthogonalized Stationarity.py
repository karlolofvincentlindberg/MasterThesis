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
    for i in range(len(cols)):
        label1 = f'{cols[i]} - ADF'
        label2 = f'{cols[i]} - KPSS'
        list2.append(label1)
        list2.append(label2)
    return list1, list2


# Checking stationarity for our computed ratios
# Reading in dataset to get columns and labels for results matrix
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
data = pd.read_excel(filename, index_col=0, engine=None,
                     sheet_name=ticker+" "+frequency)
cols = data.columns
labels, column_list = create_lists(cols)

# Checking stationarity for our computed ratios - orthogonalized data
# Reading in dataset to get columns and labels for results matrix
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
data = pd.read_excel(filename, index_col=0, engine=None)
colsss = data.columns
labels, column_list = create_lists(colsss)

# Creating dataframe to hold our results
dfmat_ort = pd.DataFrame(data=None, index=tickers, columns=column_list)

# Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe
for k in range(len(colsss)):
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
        data = pd.read_excel(filename, index_col=0, engine=None)
        result = adfuller(data.iloc[1:, k])
        r = i
        c = 2*k
        if result[1] > 0.05:
            dfmat_ort.iloc[r, c] = "No"
        else:
            dfmat_ort.iloc[r, c] = "Yes"

# KPSS test - adding results to matrix and dataframe
for k in range(len(colsss)):
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
        data = pd.read_excel(filename, index_col=0, engine=None)
        kpss_stat, p_value, lags, critical_values = kpss(data.iloc[1:, k])
        r = i
        c = 2*k+1
        if p_value < 0.05:
            dfmat_ort.iloc[r, c] = "No"
        else:
            dfmat_ort.iloc[r, c] = "Yes"

dfmat_ort.to_excel(f"{directory}/Stationarity/OrtResultsStationarity.xlsx")

# Monthly
df_ort = dfmat_ort.transpose()
df_ort = df_ort.drop(['Log Return - ADF', 'Log Return - KPSS'])
with open(directory+'/LatexTables/Stationarities/Stationarity_ort.tex', 'w') as tf:
    tf.write(df_ort.to_latex())
