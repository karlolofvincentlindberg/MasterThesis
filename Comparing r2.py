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

# Saving the file


def save_xls(list_dfs, labels, xls_path):
    with ExcelWriter(xls_path)as writer:
        for i in range(len(list_dfs)):
            list_dfs[i].to_excel(writer, sheet_name=labels[i])

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


# First linear regression
# R_t = B*S_t-1% + B*A_t-1
lin_one = "R_t = B*S_t-1% + B*dA_t-1"


def regression_one(df, lag):
    x_attn = np.array(df.iloc[1:-lag, 1]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, 2]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, -3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y, x)
    return beta, t_stat, rsq


# Second Linear Regression
# R_t = B*deltaS_t-1 + B*deltaA_t-1
lin_two = "R_t = B*deltaS_t-1 + B*deltaA_t-1"


def regression_two(df, lag):
    x_attn = np.array(df.iloc[1:-lag, -5]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, -4]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, -3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y, x)
    return beta, t_stat, rsq


# Third linear regression
# R_t = B*%deltaS_t-1% + B*%deltaA_t-1
lin_three = "R_t = B*%deltaS_t-1% + B*%deltaA_t-1"


def regression_three(df, lag):
    x_attn = np.array(df.iloc[1:-lag, -2]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, -1]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, -3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y, x)
    return beta, t_stat, rsq


# First linear regression
# R_t = B*S_t-1% + B*A_t-1
lin_one = "R_t = B*S_t% + B*dA_t"


def regression_one_comp(df, lag):
    x_attn = np.array(df.iloc[1:, 1]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:, 2]).reshape((-1, 1))
    y = np.array(df.iloc[1:, -3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y, x)
    return beta, t_stat, rsq


# Second Linear Regression
# R_t = B*deltaS_t-1 + B*deltaA_t-1
lin_two = "R_t = B*deltaS_t + B*deltaA_t"


def regression_two_comp(df, lag):
    x_attn = np.array(df.iloc[1:, -5]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:, -4]).reshape((-1, 1))
    y = np.array(df.iloc[1:, -3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y, x)
    return beta, t_stat, rsq


# Third linear regression
# R_t = B*%deltaS_t-1% + B*%deltaA_t-1
lin_three = "R_t = B*%deltaS_t% + B*%deltaA_t"


def regression_three_comp(df, lag):
    x_attn = np.array(df.iloc[1:, -2]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:, -1]).reshape((-1, 1))
    y = np.array(df.iloc[1:, -3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y, x)
    return beta, t_stat, rsq

# Presenting results if regressing both attention and sentiment


def show_results(bs, tvals, r2, freq, tick):
    r_2 = round(r2, 4)
    b_cons = round(bs[0], 4)
    b_a = round(bs[-2], 4)
    b_s = round(bs[-1], 4)
    t_cons = round(tvals[0], 4)
    t_a = round(tvals[-2], 4)
    t_s = round(tvals[-1], 4)
    print(f"{tick}: {freq} B_cons: {b_cons}, t_stat:{t_cons}")
    print(f"{tick}: {freq} B_a: {b_a}, t_stat:{t_a}")
    print(f"{tick}: {freq} B_s: {b_s}, t_stat:{t_s}")
    print(f"{tick}: {freq} R^2: {r_2}")

# Presenting results if regressing only sentiment


def show_results_se(bs, tvals, r2, freq, tick):
    r_2 = round(r2, 4)
    b_cons = round(bs[0], 4)
    b_s = round(bs[-1], 4)
    t_cons = round(tvals[0], 4)
    t_s = round(tvals[-1], 4)
    print(f"{tick}: {freq} B_cons: {b_cons}, t_stat:{t_cons}")
    print(f"{tick}: {freq} B_s: {b_s}, t_stat:{t_s}")
    print(f"{tick}: {freq} R^2: {r_2}")

# Presenting results if regressing only attention


def show_results_at(bs, tvals, r2, freq, tick):
    r_2 = round(r2, 4)
    b_cons = round(bs[0], 4)
    b_a = round(bs[-1], 4)
    t_cons = round(tvals[0], 4)
    t_a = round(tvals[-1], 4)
    print(f"{tick}: {freq} B_cons: {b_cons}, t_stat:{t_cons}")
    print(f"{tick}: {freq} B_a: {b_a}, t_stat:{t_a}")
    print(f"{tick}: {freq} R^2: {r_2}")


# Listing our frequencies & intervals
frequencies = ["Daily", "Weekly", "Monthly"]
intervals = ["Daily", "Daily", "Weekly", "Weekly", "Monthly", "Monthly"]

# Creating our index list
index_list = ["Proxy"]
for i in range(len(tickers)):
    index_list.append(tickers[i])
    index_list.append(" ")

# First regression
df_lm1 = pd.DataFrame(index=index_list, columns=intervals)

# Second regression
df_lm2 = pd.DataFrame(index=index_list, columns=intervals)

# Third regression
df_lm3 = pd.DataFrame(index=index_list, columns=intervals)


r_scores = []
labels = []
markers = ['ro', 'rv', 'rs', 'bo', 'bv', 'bs', 'go', 'gv', 'gs']

# First linear regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm1.iloc[0, 2*j] = "ATTN"
    df_lm1.iloc[0, 2*j+1] = "SENT"
    r2s = []
    label = []
    print(frequency+": "+lin_one)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        betas, t_stats, r2 = regression_one(data, 1)
        df_lm1.iloc[2*i+1, 2*j] = f"{round(betas[1]*10**3,2)}"
        df_lm1.iloc[2*i+1, 2*j+1] = f"{round(betas[2]*10**3,2)}"
        df_lm1.iloc[2*i+2, 2*j] = f"({round(t_stats[1],2)})"
        df_lm1.iloc[2*i+2, 2*j+1] = f"({round(t_stats[2],2)})"
        r2s.append(r2)
        label.append(f"{ticker}")
    r_scores.append(r2s)
    labels.append(label)

# Second linear regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm2.iloc[0, 2*j] = "ATTN"
    df_lm2.iloc[0, 2*j+1] = "SENT"
    r2s = []
    label = []
    print(frequency+": "+lin_three)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        betas, t_stats, r2 = regression_two(data, 1)
        df_lm2.iloc[2*i+1, 2*j] = f"{round(betas[1]*10**3,2)}"
        df_lm2.iloc[2*i+1, 2*j+1] = f"{round(betas[2]*10**3,2)}"
        df_lm2.iloc[2*i+2, 2*j] = f"({round(t_stats[1],2)})"
        df_lm2.iloc[2*i+2, 2*j+1] = f"({round(t_stats[2],2)})"
        r2s.append(r2)
        label.append(f"{ticker}")
    r_scores.append(r2s)
    labels.append(label)

# Third linear regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm3.iloc[0, 2*j] = "ATTN"
    df_lm3.iloc[0, 2*j+1] = "SENT"
    r2s = []
    label = []
    print(frequency+": "+lin_two)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        betas, t_stats, r2 = regression_three(data, 1)
        df_lm3.iloc[2*i+1, 2*j] = f"{round(betas[1]*10**3,2)}"
        df_lm3.iloc[2*i+1, 2*j+1] = f"{round(betas[2]*10**3,2)}"
        df_lm3.iloc[2*i+2, 2*j] = f"({round(t_stats[1],2)})"
        df_lm3.iloc[2*i+2, 2*j+1] = f"({round(t_stats[2],2)})"
        r2s.append(r2)
        label.append(f"{ticker}")
    r_scores.append(r2s)
    labels.append(label)

'''
#Plotting
plt.figure(figsize=(8,12))
for i in range(len(r_scores)):
    plt.plot(labels[i],r_scores[i],markers[i])

plt.legend(['Model1-Daily','Model1-Weekly','Model1-Monthly','Model2-Daily','Model2-Weekly','Model2-Monthly','Model5-Daily','Model3-Weekly','Model3-Monthly'])
plt.xlabel('Ticker & Frequency')
plt.ylabel('R2')
plt.show()
'''

r_score = []
labelss = []
markers = ['ro', 'rv', 'rs', 'bo', 'bv', 'bs', 'go', 'gv', 'gs']

# First linear regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm1.iloc[0, 2*j] = "ATTN"
    df_lm1.iloc[0, 2*j+1] = "SENT"
    r2s = []
    label = []
    print(frequency+": "+lin_one)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        betas, t_stats, r2 = regression_one_comp(data, 1)
        r2s.append(r2)
        label.append(f"{ticker} comp")
    r_score.append(r2s)
    labelss.append(label)

# Second linear regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm2.iloc[0, 2*j] = "ATTN"
    df_lm2.iloc[0, 2*j+1] = "SENT"
    r2s = []
    label = []
    print(frequency+": "+lin_three)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        betas, t_stats, r2 = regression_two_comp(data, 1)
        r2s.append(r2)
        label.append(f"{ticker} comp")
    r_score.append(r2s)
    labelss.append(label)

# Third linear regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm3.iloc[0, 2*j] = "ATTN"
    df_lm3.iloc[0, 2*j+1] = "SENT"
    r2s = []
    label = []
    print(frequency+": "+lin_two)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        betas, t_stats, r2 = regression_three_comp(data, 1)
        r2s.append(r2)
        label.append(f"{ticker} comp")
    r_score.append(r2s)
    labelss.append(label)

plt.figure(figsize=(8, 12))
for i in range(len(r_scores)):
    plt.plot(labelss[i], r_score[i], markers[i])

plt.legend(['Model1-Daily', 'Model1-Weekly', 'Model1-Monthly', 'Model2-Daily',
           'Model2-Weekly', 'Model2-Monthly', 'Model5-Daily', 'Model3-Weekly', 'Model3-Monthly'])
plt.xlabel('Ticker & Frequency')
plt.ylabel('R2 COMP')
plt.show()
