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


# Listing our frequencies & intervals
frequencies = ["Daily", "Monthly"]
intervals = ["Daily", "Daily", "Monthly", "Monthly"]

# Creating our index list
index_list = ["Proxy"]
for i in range(len(tickers)):
    index_list.append(tickers[i])
    index_list.append(" ")

# Fama French
# First regression
df_lm1_ff = pd.DataFrame(index=index_list, columns=intervals)

# Second regression
df_lm2_ff = pd.DataFrame(index=index_list, columns=intervals)

# Third regression
df_lm3_ff = pd.DataFrame(index=index_list, columns=intervals)

# First regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm1_ff.iloc[0, 2*j] = "ATTN"
    df_lm1_ff.iloc[0, 2*j+1] = "SENT"
    print(frequency+": "+lin_one)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/Frenched/"+ticker+"5Ratios.xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        betas, t_stats, r2 = regression_one(data, 1)
        df_lm1_ff.iloc[2*i+1, 2*j] = f"{round(betas[1]*10**3,2)}"
        df_lm1_ff.iloc[2*i+1, 2*j+1] = f"{round(betas[2]*10**3,2)}"
        df_lm1_ff.iloc[2*i+2, 2*j] = f"({round(t_stats[1],2)})"
        df_lm1_ff.iloc[2*i+2, 2*j+1] = f"({round(t_stats[2],2)})"

# Second regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm2_ff.iloc[0, 2*j] = "ATTN"
    df_lm2_ff.iloc[0, 2*j+1] = "SENT"
    print(frequency+": "+lin_three)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/Frenched/"+ticker+"5Ratios.xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        betas, t_stats, r2 = regression_two(data, 1)
        df_lm2_ff.iloc[2*i+1, 2*j] = f"{round(betas[1]*10**3,2)}"
        df_lm2_ff.iloc[2*i+1, 2*j+1] = f"{round(betas[2]*10**3,2)}"
        df_lm2_ff.iloc[2*i+2, 2*j] = f"({round(t_stats[1],2)})"
        df_lm2_ff.iloc[2*i+2, 2*j+1] = f"({round(t_stats[2],2)})"

# Third regression model
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_lm3_ff.iloc[0, 2*j] = "ATTN"
    df_lm3_ff.iloc[0, 2*j+1] = "SENT"
    print(frequency+": "+lin_two)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/Frenched/"+ticker+"5Ratios.xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        betas, t_stats, r2 = regression_three(data, 1)
        df_lm3_ff.iloc[2*i+1, 2*j] = f"{round(betas[1]*10**3,2)}"
        df_lm3_ff.iloc[2*i+1, 2*j+1] = f"{round(betas[2]*10**3,2)}"
        df_lm3_ff.iloc[2*i+2, 2*j] = f"({round(t_stats[1],2)})"
        df_lm3_ff.iloc[2*i+2, 2*j+1] = f"({round(t_stats[2],2)})"

# Saving our first linear model to late
with open(directory+'/LatexTables/Regressions/Frenched/lm1_ff5.tex', 'w') as tf:
    tf.write(df_lm1_ff.to_latex())

# Saving our second linear model to latex
with open(directory+'/LatexTables/Regressions/Frenched/lm2_ff5.tex', 'w') as tf:
    tf.write(df_lm2_ff.to_latex())

# Saving our third linear model to latex
with open(directory+'/LatexTables/Regressions/Frenched/lm3_ff5.tex', 'w') as tf:
    tf.write(df_lm3_ff.to_latex())
