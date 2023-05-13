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

# Linear
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import linear_model

from datetime import date
from pandas import ExcelWriter

# Dividing the data
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"
tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX"]
proxies = ['TC', 'TPC', 'TNC', 'NC', 'NPC', 'NNC', 'VMP', 'PCR', 'SVI', 'MTR']

# Regression model


def regression(y, x):
    x = sm.add_constant(x, prepend=True)
    model = sm.OLS(y, x)
    model_fit = model.fit()
    betas = model_fit.params
    t_stats = model_fit.tvalues
    r2 = model_fit.rsquared
    # print(model_fit.summary())
    return betas, t_stats, r2

# Function to read in our raw data


def read_data(r, freq):
    file = directory+"/18 CSVs - Final/"+tickers[r]+" - "+freq+".csv"
    data = pd.read_csv(file, index_col=0, engine=None)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data = data.set_index('Date')
    data['MTR'] = data['TV']/data['SHOUT']
    return data


def get_res(data):
    df = data.loc[:, :]
    df['lag'] = np.nan
    for i in range(1, len(df)):
        df.iloc[i, -1] = df.iloc[i-1, 0]
    df = df.iloc[1:, :]
    bs, ts, r = regression(df['Price'], df['lag'])
    alpha = bs[0]
    beta = bs[1]
    res = df.loc[:, 'Price']-alpha-beta*df.loc[:, 'lag']
    return res


# Our frequencies
frequencies = ["Daily", "Weekly", "Monthly"]

# INdex list
index_list = []
for i in range(len(proxies)):
    index_list.append(proxies[i])
    index_list.append("")


# Function to get the regression results from differenced raw proxies t


def dataframe(frequency):
    df = pd.DataFrame(columns=tickers, index=index_list)
    for i in range(len(tickers)):
        data = read_data(i, frequency)
        res = get_res(data.loc[:, :])
        data = data.diff()
        data['Res'] = res
        data = data.dropna()
        data = data[['Res']+proxies]
        print(data.head())
        for j in range(len(proxies)):
            x = 10**3
            if j == 6:
                x = 0.1
            if j == 7:
                x = 1
            if j == 9:
                x = 0.1
            proxy = proxies[j]
            bs, ts, r = regression(data['Res'], data[proxy])
            beta = f"{round(bs[1]*x,2)}"
            t_stat = f"({round(ts[1],2)})"
            df.iloc[2*j, i] = beta
            df.iloc[2*j+1, i] = t_stat
    return df

# Function to get the regression results from differenced raw proxies t-1


def dataframe_lag(frequency):
    df = pd.DataFrame(columns=tickers, index=index_list)
    for i in range(len(tickers)):
        data = read_data(i, frequency)
        res = get_res(data.loc[:, :])
        data = data.diff()
        data['Res'] = res
        data['Res Lag'] = np.nan
        for o in range(len(data)-1):
            data.iloc[o, -1] = data.iloc[o+1, -2]
        data = data.dropna()
        data = data[['Res Lag']+proxies]
        print(data.head())
        for j in range(len(proxies)):
            x = 10**3
            if j == 6:
                x = 0.1
            if j == 7:
                x = 1
            if j == 9:
                x = 0.1
            proxy = proxies[j]
            bs, ts, r = regression(data['Res Lag'], data[proxy])
            beta = f"{round(bs[1]*x,2)}"
            t_stat = f"({round(ts[1],2)})"
            df.iloc[2*j, i] = beta
            df.iloc[2*j+1, i] = t_stat
    return df


# Creating our residual regressions tables at t = t
raw_t_d = dataframe("Daily")
raw_t_w = dataframe("Weekly")
raw_t_m = dataframe("Monthly")

# Creating our residual regressions tables at t = t-1
raw_t1_d = dataframe_lag("Daily")
raw_t1_w = dataframe_lag("Weekly")
raw_t1_m = dataframe_lag("Monthly")

# Saving our residuals regression table at t= t
# Daily
with open(directory+'/LatexTables/Regressions/raw_t_d.tex', 'w') as tf:
    tf.write(raw_t_d.to_latex())

# Weekly
with open(directory+'/LatexTables/Regressions/raw_t_w.tex', 'w') as tf:
    tf.write(raw_t_w.to_latex())

# Monthly
with open(directory+'/LatexTables/Regressions/raw_t_m.tex', 'w') as tf:
    tf.write(raw_t_m.to_latex())

# Saving our residuals regression table at t = t-1
# Daily
with open(directory+'/LatexTables/Regressions/raw_t1_d.tex', 'w') as tf:
    tf.write(raw_t1_d.to_latex())

# Weekly
with open(directory+'/LatexTables/Regressions/raw_t1_w.tex', 'w') as tf:
    tf.write(raw_t1_w.to_latex())

# Monthly
with open(directory+'/LatexTables/Regressions/raw_t1_m.tex', 'w') as tf:
    tf.write(raw_t1_m.to_latex())
