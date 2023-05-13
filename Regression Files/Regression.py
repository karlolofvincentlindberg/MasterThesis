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

ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/RatioDATA/"+ticker+".xlsx"

data = pd.read_excel(filename, index_col=0, engine=None,
                     sheet_name=ticker+" "+frequency)

lag = 1

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

# Transforming dataset and creating residuals


def data_res(df):
    data = pd.DataFrame(df)
    data['Price lag'] = np.nan
    for i in range(1, len(data)):
        data.iloc[i, -1] = data.iloc[i-1, 0]
    bs, ts, r2 = regression(data.iloc[1:, 0], data.iloc[1:, -1])
    alpha = bs[0]
    beta = bs[1]
    data['Res'] = data['Price'] - alpha - beta*data['Price lag']
    data['Return'] = data['Res']
    data = data.drop(['Price', 'Price lag', 'Res'], axis=1)
    return data


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

# Orthoganalized regression
ortho_reg = pd.DataFrame(index=index_list, columns=[
                         "Model 1", "", "Model 2", "", "Model 3", ""])

# Residual daily regression
res_d_reg = pd.DataFrame(index=index_list, columns=[
                         "Model 1", "", "Model 2", "", "Model 3", ""])

# Residual weekly regression
res_w_reg = pd.DataFrame(index=index_list, columns=[
                         "Model 1", "", "Model 2", "", "Model 3", ""])

# Residual monthly regression
res_m_reg = pd.DataFrame(index=index_list, columns=[
                         "Model 1", "", "Model 2", "", "Model 3", ""])

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

# Saving our first linear model to latex
with open(directory+'/LatexTables/Regressions/Linear/lm1.tex', 'w') as tf:
    tf.write(df_lm1.to_latex())

# Saving our second linear model to latex
with open(directory+'/LatexTables/Regressions/Linear/lm2.tex', 'w') as tf:
    tf.write(df_lm2.to_latex())

# Saving our third linear model to latex
with open(directory+'/LatexTables/Regressions/Linear/lm3.tex', 'w') as tf:
    tf.write(df_lm3.to_latex())

# Plotting r2s
plt.figure(figsize=(8, 12))
for i in range(len(r_scores)):
    plt.plot(labels[i], r_scores[i], markers[i])
plt.legend(['Model1-Daily', 'Model1-Weekly', 'Model1-Monthly', 'Model2-Daily',
           'Model2-Weekly', 'Model2-Monthly', 'Model5-Daily', 'Model3-Weekly', 'Model3-Monthly'])
plt.xlabel('R2')
plt.ylabel('Ticker & Frequency')
plt.show()

# Regressing using orthogonalized data
print("Orthogonalized")
print(lin_one)
ortho_reg.iloc[0, :] = ['ATTN', 'SENT', 'ATTN', 'SENT', 'ATTN', 'SENT']
for i in range(len(tickers)):
    ticker = tickers[i]
    filename = directory+"/Orthogonalized/"+ticker+"Ratios.xlsx"
    data = pd.read_excel(filename, index_col=0, engine=None)
    b1, t_sta1, r1 = regression_one(data, 1)
    b2, t_sta2, r2 = regression_two(data, 1)
    b3, t_sta3, r3 = regression_three(data, 1)
    ortho_reg.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    ortho_reg.iloc[2*i+2, 0] = f"({round(t_sta1[1],2)})"
    ortho_reg.iloc[2*i+1, 1] = f"{round(b1[2]*10**3,2)}"
    ortho_reg.iloc[2*i+2, 1] = f"({round(t_sta1[2],2)})"
    ortho_reg.iloc[2*i+1, 2] = f"{round(b2[1]*10**3,2)}"
    ortho_reg.iloc[2*i+2, 2] = f"({round(t_sta2[1],2)})"
    ortho_reg.iloc[2*i+1, 3] = f"{round(b2[2]*10**3,2)}"
    ortho_reg.iloc[2*i+2, 3] = f"({round(t_sta2[2],2)})"
    ortho_reg.iloc[2*i+1, 4] = f"{round(b3[1]*10**3,2)}"
    ortho_reg.iloc[2*i+2, 4] = f"({round(t_sta3[1],2)})"
    ortho_reg.iloc[2*i+1, 5] = f"{round(b3[2]*10**3,2)}"
    ortho_reg.iloc[2*i+2, 5] = f"({round(t_sta3[2],2)})"

# Saving our orthoganal regression table
with open(directory+'/LatexTables/Regressions/orthoreg.tex', 'w') as tf:
    tf.write(ortho_reg.to_latex())

# Regressing using residuals
# P_t = a + B*P_t-1 +res
# res = p_t - a - b*P_t-1
dfs = []
tickers_lab = []

# Residuals daily
res_d_reg.iloc[0, :] = ['ATTN', 'SENT', 'ATTN', 'SENT', 'ATTN', 'SENT']
for i in range(len(tickers)):
    frequency = "Daily"
    ticker = tickers[i]
    filename = directory+"/RatioDATA/"+ticker+".xlsx"
    df = pd.read_excel(filename, index_col=0, engine=None,
                       sheet_name=ticker+" "+frequency)
    data = data_res(df)
    dfs.append(data)
    tickers_lab.append(f"{ticker} {frequency}")
    b1, t_sta1, r1 = regression_one(data, 1)
    b2, t_sta2, r2 = regression_two(data, 1)
    b3, t_sta3, r3 = regression_three(data, 1)
    res_d_reg.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    res_d_reg.iloc[2*i+2, 0] = f"({round(t_sta1[1],2)})"
    res_d_reg.iloc[2*i+1, 1] = f"{round(b2[1]*10**3,2)}"
    res_d_reg.iloc[2*i+2, 1] = f"({round(t_sta1[2],2)})"
    res_d_reg.iloc[2*i+1, 2] = f"{round(b2[1]*10**3,2)}"
    res_d_reg.iloc[2*i+2, 2] = f"({round(t_sta2[1],2)})"
    res_d_reg.iloc[2*i+1, 3] = f"{round(b2[2]*10**3,2)}"
    res_d_reg.iloc[2*i+2, 3] = f"({round(t_sta2[2],2)})"
    res_d_reg.iloc[2*i+1, 4] = f"{round(b3[1]*10**3,2)}"
    res_d_reg.iloc[2*i+2, 4] = f"({round(t_sta3[1],2)})"
    res_d_reg.iloc[2*i+1, 5] = f"{round(b3[2]*10**3,2)}"
    res_d_reg.iloc[2*i+2, 5] = f"({round(t_sta3[2],2)})"

# Residuals weekly
res_w_reg.iloc[0, :] = ['ATTN', 'SENT', 'ATTN', 'SENT', 'ATTN', 'SENT']
for i in range(len(tickers)):
    frequency = "Weekly"
    ticker = tickers[i]
    filename = directory+"/RatioDATA/"+ticker+".xlsx"
    df = pd.read_excel(filename, index_col=0, engine=None,
                       sheet_name=ticker+" "+frequency)
    data = data_res(df)
    dfs.append(data)
    tickers_lab.append(f"{ticker} {frequency}")
    b1, t_sta1, r1 = regression_one(data, 1)
    b2, t_sta2, r2 = regression_two(data, 1)
    b3, t_sta3, r3 = regression_three(data, 1)
    res_w_reg.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    res_w_reg.iloc[2*i+2, 0] = f"({round(t_sta1[1],2)})"
    res_w_reg.iloc[2*i+1, 1] = f"{round(b2[1]*10**3,2)}"
    res_w_reg.iloc[2*i+2, 1] = f"({round(t_sta1[2],2)})"
    res_w_reg.iloc[2*i+1, 2] = f"{round(b2[1]*10**3,2)}"
    res_w_reg.iloc[2*i+2, 2] = f"({round(t_sta2[1],2)})"
    res_w_reg.iloc[2*i+1, 3] = f"{round(b2[2]*10**3,2)}"
    res_w_reg.iloc[2*i+2, 3] = f"({round(t_sta2[2],2)})"
    res_w_reg.iloc[2*i+1, 4] = f"{round(b3[1]*10**3,2)}"
    res_w_reg.iloc[2*i+2, 4] = f"({round(t_sta3[1],2)})"
    res_w_reg.iloc[2*i+1, 5] = f"{round(b3[2]*10**3,2)}"
    res_w_reg.iloc[2*i+2, 5] = f"({round(t_sta3[2],2)})"

# Residuals monthly
res_m_reg.iloc[0, :] = ['ATTN', 'SENT', 'ATTN', 'SENT', 'ATTN', 'SENT']
for i in range(len(tickers)):
    frequency = "Monthly"
    ticker = tickers[i]
    filename = directory+"/RatioDATA/"+ticker+".xlsx"
    df = pd.read_excel(filename, index_col=0, engine=None,
                       sheet_name=ticker+" "+frequency)
    data = data_res(df)
    dfs.append(data)
    tickers_lab.append(f"{ticker} {frequency}")
    b1, t_sta1, r1 = regression_one(data, 1)
    b2, t_sta2, r2 = regression_two(data, 1)
    b3, t_sta3, r3 = regression_three(data, 1)
    res_m_reg.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    res_m_reg.iloc[2*i+2, 0] = f"({round(t_sta1[1],2)})"
    res_m_reg.iloc[2*i+1, 1] = f"{round(b2[1]*10**3,2)}"
    res_m_reg.iloc[2*i+2, 1] = f"({round(t_sta1[2],2)})"
    res_m_reg.iloc[2*i+1, 2] = f"{round(b2[1]*10**3,2)}"
    res_m_reg.iloc[2*i+2, 2] = f"({round(t_sta2[1],2)})"
    res_m_reg.iloc[2*i+1, 3] = f"{round(b2[2]*10**3,2)}"
    res_m_reg.iloc[2*i+2, 3] = f"({round(t_sta2[2],2)})"
    res_m_reg.iloc[2*i+1, 4] = f"{round(b3[1]*10**3,2)}"
    res_m_reg.iloc[2*i+2, 4] = f"({round(t_sta3[1],2)})"
    res_m_reg.iloc[2*i+1, 5] = f"{round(b3[2]*10**3,2)}"
    res_m_reg.iloc[2*i+2, 5] = f"({round(t_sta3[2],2)})"

# Saving our residuals regression table
# Daily
with open(directory+'/LatexTables/Regressions/Residual/resid_d.tex', 'w') as tf:
    tf.write(res_d_reg.to_latex())

# Weekly
with open(directory+'/LatexTables/Regressions/Residual/resid_w.tex', 'w') as tf:
    tf.write(res_w_reg.to_latex())

# Monthly
with open(directory+'/LatexTables/Regressions/Residual/resid_m.tex', 'w') as tf:
    tf.write(res_m_reg.to_latex())

# Saving our residual data
for i in range(len(dfs)):
    filepath = directory+"/ResData/"+tickers_lab[i]+".xlsx"
    dfs[i].to_excel(filepath)
    print(i)


# Creating x variables when using them in exponential models
def create_xvars(x1, x2):
    xa = []
    xb = x2
    for i in range(len(x1)):
        x = math.log(abs(x1[i]))
        xa.append(x)
    x_all = np.hstack((np.array(xa).reshape((-1, 1)), xb))
    return x_all


# Introducing our exponential models
# First exponential model
# Option 1
ex_one_one = "log R_t = B*ln(|A_t-1|) + B*S_t-1"
# Option 2
ex_one_two = "log R_t = B*ln(|S_t-1|) + B*A_t-1"
# Option 3
ex_one_three = "log R_t = ln(C) + B*A_t-1*S_t-1"


def expo_one(df, lag):
    x_attn = np.array(df.iloc[1:-lag, 1]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, 2]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, 3])
    x1 = create_xvars(x_attn, x_sent)
    x2 = create_xvars(x_sent, x_attn)
    x3 = x_attn*x_sent
    b1, t_s1, rsq1 = regression(y, x1)
    b2, t_s2, rsq2 = regression(y, x2)
    b3, t_s3, rsq3 = regression(y, x3)
    return b1, b2, b3, t_s1, t_s2, t_s3


# First exponential model
# Option 1
ex_two_one = "log R_t = B*ln(|deltaA_t-1|) + B*deltaS_t-1"
# Option 2
ex_two_two = "log R_t = B*ln(|deltaS_t-1|) + B*deltaA_t-1"
# Option 3
ex_two_three = "log R_t = ln(C) + B*deltaA_t-1*deltaS_t-1"


def expo_two(df, lag):
    x_attn = np.array(df.iloc[1:-lag, -5]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, -4]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, 3])
    x1 = create_xvars(x_attn, x_sent)
    x2 = create_xvars(x_sent, x_attn)
    x3 = x_attn*x_sent
    b1, t_s1, rsq1 = regression(y, x1)
    b2, t_s2, rsq2 = regression(y, x2)
    b3, t_s3, rsq3 = regression(y, x3)
    return b1, b2, b3, t_s1, t_s2, t_s3


# Third exponential model
# Option 1
ex_three_one = "log R_t = B*ln(|%deltaA_t-1|) + B*%deltaS_t-1"
# Option 2
ex_three_two = "log R_t = B*ln(|%deltaS_t-1|) + B*%deltaA_t-1"
# Option 3
ex_three_three = "log R_t = ln(C) + B*%deltaA_t-1*%deltaS_t-1"


def expo_three(df, lag):
    x_attn = np.array(df.iloc[1:-lag, -2]).reshape((-1, 1))
    x_sent = np.array(df.iloc[1:-lag, -1]).reshape((-1, 1))
    y = np.array(df.iloc[lag+1:, 3])
    x1 = create_xvars(x_attn, x_sent)
    x2 = create_xvars(x_sent, x_attn)
    x3 = x_attn*x_sent
    b1, t_s1, rsq1 = regression(y, x1)
    b2, t_s2, rsq2 = regression(y, x2)
    b3, t_s3, rsq3 = regression(y, x3)
    return b1, b2, b3, t_s1, t_s2, t_s3


# First exponential models
df_em1_1 = pd.DataFrame(index=index_list, columns=intervals)
df_em1_2 = pd.DataFrame(index=index_list, columns=intervals)
df_em1_3 = pd.DataFrame(index=index_list, columns=intervals)

# Second exponential models
df_em2_1 = pd.DataFrame(index=index_list, columns=intervals)
df_em2_2 = pd.DataFrame(index=index_list, columns=intervals)
df_em2_3 = pd.DataFrame(index=index_list, columns=intervals)

# Third exponential models
df_em3_1 = pd.DataFrame(index=index_list, columns=intervals)
df_em3_2 = pd.DataFrame(index=index_list, columns=intervals)
df_em3_3 = pd.DataFrame(index=index_list, columns=intervals)

# Main exponential model 1
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_em1_1.iloc[0, :] = [
        "ln(ATTN)", "SENT", "ln(ATTN)", "SENT", "ln(ATTN)", "SENT"]
    df_em1_2.iloc[0, :] = [
        "ATTN", "ln(SENT)", "ATTN", "ln(SENT)", "ATTN", "ln(SENT)"]
    df_em1_3.iloc[0, :] = [
        "ln(C)", "ATTN*SENT", "ln(C)", "ATTN*SENT", "ln(C)", "ATTN*SENT"]
    print(frequency+": "+lin_three)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        b1, b2, b3, t_s1, t_s2, t_s3 = expo_one(data, 1)
        print(b3)
        df_em1_1.iloc[2*i+1, 2*j] = f"{round(b1[1]*10**3,2)}"
        df_em1_1.iloc[2*i+1, 2*j+1] = f"{round(b1[2]*10**3,2)}"
        df_em1_1.iloc[2*i+2, 2*j] = f"({round(t_s1[1],2)})"
        df_em1_1.iloc[2*i+2, 2*j+1] = f"({round(t_s1[2],2)})"
        df_em1_2.iloc[2*i+1, 2*j+1] = f"{round(b2[1]*10**3,2)}"
        df_em1_2.iloc[2*i+1, 2*j] = f"{round(b2[2]*10**3,2)}"
        df_em1_2.iloc[2*i+2, 2*j+1] = f"({round(t_s2[1],2)})"
        df_em1_2.iloc[2*i+2, 2*j] = f"({round(t_s2[2],2)})"
        df_em1_3.iloc[2*i+1, 2*j+1] = f"{round(b3[1]*10**3,2)}"
        df_em1_3.iloc[2*i+1, 2*j] = f"{round(b3[0]*10**3,2)}"
        df_em1_3.iloc[2*i+2, 2*j+1] = f"({round(t_s3[1],2)})"
        df_em1_3.iloc[2*i+2, 2*j] = f"({round(t_s3[0],2)})"

# Main exponential model 2
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_em2_1.iloc[0, :] = [
        "ln(ATTN)", "SENT", "ln(ATTN)", "SENT", "ln(ATTN)", "SENT"]
    df_em2_2.iloc[0, :] = [
        "ATTN", "ln(SENT)", "ATTN", "ln(SENT)", "ATTN", "ln(SENT)"]
    df_em2_3.iloc[0, :] = [
        "ln(C)", "ATTN*SENT", "ln(C)", "ATTN*SENT", "ln(C)", "ATTN*SENT"]
    print(frequency+": "+lin_three)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        b1, b2, b3, t_s1, t_s2, t_s3 = expo_two(data, 1)
        print(b3)
        df_em2_1.iloc[2*i+1, 2*j] = f"{round(b1[1]*10**3,2)}"
        df_em2_1.iloc[2*i+1, 2*j+1] = f"{round(b1[2]*10**3,2)}"
        df_em2_1.iloc[2*i+2, 2*j] = f"({round(t_s1[1],2)})"
        df_em2_1.iloc[2*i+2, 2*j+1] = f"({round(t_s1[2],2)})"
        df_em2_2.iloc[2*i+1, 2*j+1] = f"{round(b2[1]*10**3,2)}"
        df_em2_2.iloc[2*i+1, 2*j] = f"{round(b2[2]*10**3,2)}"
        df_em2_2.iloc[2*i+2, 2*j+1] = f"({round(t_s2[1],2)})"
        df_em2_2.iloc[2*i+2, 2*j] = f"({round(t_s2[2],2)})"
        df_em2_3.iloc[2*i+1, 2*j+1] = f"{round(b3[1]*10**3,2)}"
        df_em2_3.iloc[2*i+1, 2*j] = f"{round(b3[0]*10**3,2)}"
        df_em2_3.iloc[2*i+2, 2*j+1] = f"({round(t_s3[1],2)})"
        df_em2_3.iloc[2*i+2, 2*j] = f"({round(t_s3[0],2)})"

# Main exponential model 3
for j in range(len(frequencies)):
    frequency = frequencies[j]
    df_em3_1.iloc[0, :] = [
        "ln(ATTN)", "SENT", "ln(ATTN)", "SENT", "ln(ATTN)", "SENT"]
    df_em3_2.iloc[0, :] = [
        "ATTN", "ln(SENT)", "ATTN", "ln(SENT)", "ATTN", "ln(SENT)"]
    df_em3_3.iloc[0, :] = [
        "ln(C)", "ATTN*SENT", "ln(C)", "ATTN*SENT", "ln(C)", "ATTN*SENT"]
    print(frequency+": "+lin_three)
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        data = data.drop('Price', axis=1)
        b1, b2, b3, t_s1, t_s2, t_s3 = expo_three(data, 1)
        print(b3)
        df_em3_1.iloc[2*i+1, 2*j] = f"{round(b1[1]*10**3,2)}"
        df_em3_1.iloc[2*i+1, 2*j+1] = f"{round(b1[2]*10**3,2)}"
        df_em3_1.iloc[2*i+2, 2*j] = f"({round(t_s1[1],2)})"
        df_em3_1.iloc[2*i+2, 2*j+1] = f"({round(t_s1[2])})"
        df_em3_2.iloc[2*i+1, 2*j+1] = f"{round(b2[1]*10**3,2)}"
        df_em3_2.iloc[2*i+1, 2*j] = f"{round(b2[2]*10**3,2)}"
        df_em3_2.iloc[2*i+2, 2*j+1] = f"({round(t_s2[1],2)})"
        df_em3_2.iloc[2*i+2, 2*j] = f"({round(t_s2[2],2)})"
        df_em3_3.iloc[2*i+1, 2*j+1] = f"{round(b3[1]*10**3,2)}"
        df_em3_3.iloc[2*i+1, 2*j] = f"{round(b3[0]*10**3,2)}"
        df_em3_3.iloc[2*i+2, 2*j+1] = f"({round(t_s3[1],2)})"
        df_em3_3.iloc[2*i+2, 2*j] = f"({round(t_s3[0],2)})"

# Gathering all exponential models in one dataframe
dataframes = [df_em1_1, df_em1_2, df_em1_3, df_em2_1,
              df_em2_2, df_em2_3, df_em3_1, df_em3_2, df_em3_3]
datalabels = ["em1_1", "em1_2", "em1_3", "em2_1",
              "em2_2", "em2_3", "em3_1", "em3_2", "em3_3"]

for i in range(len(dataframes)):
    hint = datalabels[i]
    with open(directory+'/LatexTables/Regressions/Exponential/' + hint+'.tex', 'w') as tf:
        tf.write(dataframes[i].to_latex())


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
        filename = directory+"/Frenched/"+ticker+"Ratios.xlsx"
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
        filename = directory+"/Frenched/"+ticker+"Ratios.xlsx"
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
        filename = directory+"/Frenched/"+ticker+"Ratios.xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        betas, t_stats, r2 = regression_three(data, 1)
        df_lm3_ff.iloc[2*i+1, 2*j] = f"{round(betas[1]*10**3,2)}"
        df_lm3_ff.iloc[2*i+1, 2*j+1] = f"{round(betas[2]*10**3,2)}"
        df_lm3_ff.iloc[2*i+2, 2*j] = f"({round(t_stats[1],2)})"
        df_lm3_ff.iloc[2*i+2, 2*j+1] = f"({round(t_stats[2],2)})"

# Saving our first linear model to late
with open(directory+'/LatexTables/Regressions/Frenched/lm1_ff.tex', 'w') as tf:
    tf.write(df_lm1_ff.to_latex())

# Saving our second linear model to latex
with open(directory+'/LatexTables/Regressions/Frenched/lm2_ff.tex', 'w') as tf:
    tf.write(df_lm2_ff.to_latex())

# Saving our third linear model to latex
with open(directory+'/LatexTables/Regressions/Frenched/lm3_ff.tex', 'w') as tf:
    tf.write(df_lm3_ff.to_latex())
