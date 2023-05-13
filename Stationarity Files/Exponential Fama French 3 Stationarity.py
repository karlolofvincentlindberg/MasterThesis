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


# Exponential models
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/SavedData/Frenched/"+ticker+"Ratios.xlsx"
data = pd.read_excel(filename, index_col=0, engine=None,
                     sheet_name=ticker+" "+frequency)
colss = ['Log Return', 'ln(|ATTN|)', 'ln(|SENT|)', 'ATTN*SENT', 'ln(|deltaATTN|)', 'ln(|deltaSENT|)',
         'deltaATTN*deltaSENT', 'ln(|%deltaATTN|)', 'ln(|%deltaSENT|)', '%deltaATTN*%deltaSENT']
labelss, columns_list = create_lists(colss)


def transform_data(data):
    df = pd.DataFrame(index=data.index, columns=colss)
    for i in range(len(df)):
        df.iloc[i, 0] = data.iloc[i, 3]
        df.iloc[i, 1] = math.log(abs(data.iloc[i, 1]))
        df.iloc[i, 2] = math.log(abs(data.iloc[i, 2]))
        df.iloc[i, 3] = data.iloc[i, 1]*data.iloc[i, 2]
        df.iloc[i, 4] = math.log(abs(data.iloc[i, 4]))
        df.iloc[i, 5] = math.log(abs(data.iloc[i, 5]))
        df.iloc[i, 6] = data.iloc[i, 4]*data.iloc[i, 5]
        df.iloc[i, 7] = math.log(abs(data.iloc[i, 7]))
        df.iloc[i, 8] = math.log(abs(data.iloc[i, 8]))
        df.iloc[i, 9] = data.iloc[i, 8]*data.iloc[i, 7]
    return df


# Creating dataframe to hold our results
dfmat_log = pd.DataFrame(data=None, index=labelss, columns=columns_list)

# Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe
for k in range(len(colss)):
    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            filename = directory+"/SavedData/Frenched/"+ticker+"Ratios.xlsx"
            data = pd.read_excel(filename, index_col=0,
                                 engine=None, sheet_name=ticker+" "+frequency)
            new_data = transform_data(data)
            result = adfuller(new_data.iloc[1:, k])
            r = j*6+i
            c = 2*k
            if result[1] > 0.05:
                dfmat_log.iloc[r, c] = "No"
            else:
                dfmat_log.iloc[r, c] = "Yes"

# KPSS test - adding results to matrix and dataframe
for k in range(len(colss)):
    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            filename = directory+"/SavedData/Frenched/"+ticker+"Ratios.xlsx"
            data = pd.read_excel(filename, index_col=0,
                                 engine=None, sheet_name=ticker+" "+frequency)
            new_data = transform_data(data)
            kpss_stat, p_value, lags, critical_values = kpss(
                new_data.iloc[1:, k])
            r = j*6+i
            c = 2*k+1
            if p_value < 0.05:
                dfmat_log.iloc[r, c] = "No"
            else:
                dfmat_log.iloc[r, c] = "Yes"
dfmat_log.to_excel(f"{directory}/Stationarity/FF3LogResultsStationarity.xlsx")

# Stationarity of computed ratios for linear regressions
# Daily
df_d_log = dfmat_log.transpose().iloc[:, :6]
df_d_log.columns = tickers
with open(directory+'/LatexTables/Stationarities/Stationarity_d_log_ff.tex', 'w') as tf:
    tf.write(df_d_log.to_latex())

# Weekly
df_w_log = dfmat_log.transpose().iloc[:, 6:12]
df_w_log.columns = tickers
with open(directory+'/LatexTables/Stationarities/Stationarity_w_log_ff.tex', 'w') as tf:
    tf.write(df_w_log.to_latex())

# Monthly
df_m_log = dfmat_log.transpose().iloc[:, 12:]
df_m_log.columns = tickers
with open(directory+'/LatexTables/Stationarities/Stationarity_m_log_ff.tex', 'w') as tf:
    tf.write(df_m_log.to_latex())
