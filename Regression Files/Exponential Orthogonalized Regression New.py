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

# Listing our frequencies & intervals
frequencies = ["Daily", "Weekly", "Monthly"]
intervals = ["Daily", "Daily", "Weekly", "Weekly", "Monthly", "Monthly"]

# Creating our index list
index_list = ["Proxy"]
for i in range(len(tickers)):
    index_list.append(tickers[i])
    index_list.append(" ")


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

# Creating x variables when using them in exponential models


def create_xvars(x1, x2):
    xa = []
    xb = x2
    min = x1.min() - 1
    for i in range(len(x1)):
        x = math.log(x1[i]-min)
        xa.append(x)
    x_all = np.hstack((np.array(xa).reshape((-1, 1)), xb))
    return x_all


# Introducing our exponential models
# First exponential model
ex_one_one = "log R_t = B*ln(|A_t-1|) + B*S_t-1"


def expo_one(df, lag):
    x_attn = np.array(df.iloc[1:-lag, 1]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, 2]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, 3])
    x1 = create_xvars(x_attn, x_sent)
    b1, t_s1, rsq1 = regression(y, x1)
    return b1, t_s1


# First exponential model
ex_two_one = "log R_t = B*ln(|deltaA_t-1|) + B*deltaS_t-1"


def expo_two(df, lag):
    x_attn = np.array(df.iloc[1:-lag, -5]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, -4]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, 3])
    x1 = create_xvars(x_attn, x_sent)
    b1, t_s1, rsq1 = regression(y, x1)
    return b1, t_s1


# Third exponential model
ex_three_one = "log R_t = B*ln(|%deltaA_t-1|) + B*%deltaS_t-1"


def expo_three(df, lag):
    x_attn = np.array(df.iloc[1:-lag, -2]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, -1]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, 3])
    x1 = create_xvars(x_attn, x_sent)
    b1, t_s1, rsq1 = regression(y, x1)
    return b1, t_s1


# First exponential models
df_em1_1 = pd.DataFrame(index=index_list, columns=intervals)

# Second exponential models
df_em2_1 = pd.DataFrame(index=index_list, columns=intervals)


# Third exponential models
df_em3_1 = pd.DataFrame(index=index_list, columns=intervals)

# Main exponential model 1

frequency = "Monthly"
df_em1_1.iloc[0, :] = ["ln(ATTN)", "SENT", "ln(ATTN)",
                       "SENT", "ln(ATTN)", "SENT"]

for i in range(len(tickers)):
    ticker = tickers[i]
    filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
    data = pd.read_excel(filename, index_col=0, engine=None)
    b1, t_s1 = expo_one(data, 1)
    df_em1_1.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    df_em1_1.iloc[2*i+1, 1] = f"{round(b1[2]*10**3,2)}"
    df_em1_1.iloc[2*i+2, 0] = f"({round(t_s1[1],2)})"
    df_em1_1.iloc[2*i+2, 1] = f"({round(t_s1[2],2)})"

# Main exponential model 2
frequency = "Monthly"
df_em2_1.iloc[0, :] = [
    "ln(ATTN)", "SENT", "ln(ATTN)", "SENT", "ln(ATTN)", "SENT"]
for i in range(len(tickers)):
    ticker = tickers[i]
    filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
    data = pd.read_excel(filename, index_col=0, engine=None)
    b1, t_s1 = expo_two(data, 1)
    df_em2_1.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    df_em2_1.iloc[2*i+1, 1] = f"{round(b1[2]*10**3,2)}"
    df_em2_1.iloc[2*i+2, 0] = f"({round(t_s1[1],2)})"
    df_em2_1.iloc[2*i+2, 1] = f"({round(t_s1[2],2)})"

# Main exponential model 3

frequency = "Monthly"
df_em3_1.iloc[0, :] = [
    "ln(|ATTN|)", "SENT", "ln(|ATTN|)", "SENT", "ln(|ATTN|)", "SENT"]
for i in range(len(tickers)):
    ticker = tickers[i]
    filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
    data = pd.read_excel(filename, index_col=0, engine=None)
    b1, t_s1 = expo_three(data, 1)
    df_em3_1.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    df_em3_1.iloc[2*i+1, 1] = f"{round(b1[2]*10**3,2)}"
    df_em3_1.iloc[2*i+2, 0] = f"({round(t_s1[1],2)})"
    df_em3_1.iloc[2*i+2, 1] = f"({round(t_s1[2],2)})"

# Gathering all exponential models in one dataframe
dataframes = [df_em1_1, df_em2_1, df_em3_1]
datalabels = ["em1_1_ort", "em1_2_ort", "em1_3_ort", "em2_1_ort",
              "em2_2_ort", "em2_3_ort", "em3_1_ort", "em3_2_ort", "em3_3_ort"]

models1 = ["Model 5.1", "", "Model 5.2", "", "Model 5.3", ""]
df_join_one = pd.DataFrame(index=df_em1_1.index, columns=models1)

df_join_one.iloc[:, :2] = df_em1_1.iloc[:, 0:2]
df_join_one.iloc[:, 2:4] = df_em2_1.iloc[:, 0:2]
df_join_one.iloc[:, 4:6] = df_em3_1.iloc[:, 0:2]


with open(directory+'/LatexTables/Regressions/Exponential/Orthonewest.tex', 'w') as tf:
    tf.write(df_join_one.to_latex())
