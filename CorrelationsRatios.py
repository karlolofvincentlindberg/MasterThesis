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
frequencies = ["Daily", "Weekly", "Monthly"]

# Reading the data


def read_data(freq, i):
    ticker = tickers[i]
    filename = directory+"/SavedData/RatioDATA/"+ticker+".xlsx"
    data = pd.read_excel(filename, index_col=0, engine=None,
                         sheet_name=ticker+" "+freq)
    return data

# Getting the sentiment


def get_sent(freq, i):
    df = read_data(freq, i)
    sent = df['SENT']
    return sent

# Getting the attention


def get_attn(freq, i):
    df = read_data(freq, i)
    attn = df['ATTN']
    return attn

# Getting the return


def get_ret(freq, i):
    df = read_data(freq, i)
    ret = df['Return']
    return ret

# Getting correlation


def get_corrs(freq, i):
    r = np.array(get_ret(freq, i)[1:])
    sent = np.array(get_sent(freq, i)[:-1])
    attn = np.array(get_attn(freq, i)[:-1])
    df = pd.DataFrame()
    df['Ret'] = r
    df['SENT'] = sent
    df['ATTN'] = attn
    cor = df.corr()
    corr_attn = cor.iloc[2,0]
    corr_sent = cor.iloc[1,0]
    return corr_attn, corr_sent

# Gett n-values for each data
n_d = len(get_ret("Daily", 0)[1:])
n_w = len(get_ret("Weekly", 0)[1:])
n_m = len(get_ret("Monthly", 0)[1:])

# Getting the significance
def get_corrsig(r, freq):
    n = len(get_ret(freq, 0)[1:])
    upper = r * math.sqrt(n-2)
    lower = math.sqrt(1-r**2)
    sig = upper/lower
    return sig

r = get_ret("Daily", 0)
x = get_sent("Daily", 0)
y = get_attn("Daily", 0)

# Sentiment dataframes
sent_d = pd.DataFrame(columns=tickers)
sent_w = pd.DataFrame(columns=tickers)
sent_m = pd.DataFrame(columns=tickers)

# Attention dataframes
attn_d = pd.DataFrame(columns=tickers)
attn_w = pd.DataFrame(columns=tickers)
attn_m = pd.DataFrame(columns=tickers)

# Getting sentiments and attentions for each company and timeframe
for i in range(len(tickers)):
    ticker = tickers[i]
    sent_d[ticker] = get_sent("Daily", i)
    sent_w[ticker] = get_sent("Weekly", i)
    sent_m[ticker] = get_sent("Monthly", i)
    attn_d[ticker] = get_attn("Daily", i)
    attn_w[ticker] = get_attn("Weekly", i)
    attn_m[ticker] = get_attn("Monthly", i)

# Remove half of the values
def only_half(data):
    ind = data.index
    cols = data.columns
    matrix = np.array(data)
    size = len(data)

    res = matrix[np.tril_indices(size)]
    mat = np.ones((size, size)) * np.nan
    j = 0
    for i in range(size):
        j = j + i
        mat[i][:i+1] = res[j:j+i+1]
    df = pd.DataFrame(data=mat, index=ind, columns=cols)
    return df


# Sentiment correlations
corr_sent_d = round(only_half(sent_d.corr()), 2)
corr_sent_w = round(only_half(sent_w.corr()), 2)
corr_sent_m = round(only_half(sent_m.corr()), 2)

# Attention correlations
corr_attn_d = round(only_half(attn_d.corr()), 2)
corr_attn_w = round(only_half(attn_w.corr()), 2)
corr_attn_m = round(only_half(attn_m.corr()), 2)

index_list = [" "]+tickers+[" "]+tickers+[" "]+tickers

all_corr_sent = pd.DataFrame(columns=tickers, index=index_list)
all_corr_attn = pd.DataFrame(columns=tickers, index=index_list)

# Assembling all correlations
# Sentiment
all_corr_sent.iloc[0, :] = ["Daily"]*6
all_corr_sent.iloc[1:7, :] = corr_sent_d
all_corr_sent.iloc[7, :] = ["Weekly"]*6
all_corr_sent.iloc[8:14, :] = corr_sent_w
all_corr_sent.iloc[14, :] = ["Monthly"]*6
all_corr_sent.iloc[15:, :] = corr_sent_m

# Attention
all_corr_attn.iloc[0, :] = ["Daily"]*6
all_corr_attn.iloc[1:7, :] = corr_attn_d
all_corr_attn.iloc[7, :] = ["Weekly"]*6
all_corr_attn.iloc[8:14, :] = corr_attn_w
all_corr_attn.iloc[14, :] = ["Monthly"]*6
all_corr_attn.iloc[15:, :] = corr_attn_m

# Filling in values not existing
for i in range(len(all_corr_attn)):
    for j in range(len(tickers)):
        if pd.isna(all_corr_attn.iloc[i, j]):
            all_corr_attn.iloc[i, j] = ""
        if pd.isna(all_corr_sent.iloc[i, j]):
            all_corr_sent.iloc[i, j] = ""

#Saving to latex
with open(directory+'/LatexTables/corr_sent.tex', 'w') as tf:
    tf.write(all_corr_sent.to_latex())

with open(directory+'/LatexTables/corr_attn.tex', 'w') as tf:
    tf.write(all_corr_attn.to_latex())

#Correlation Significance 
sig_attn_d = pd.DataFrame(data = np.nan, columns = tickers, index = tickers)
sig_attn_w = pd.DataFrame(data = np.nan, columns = tickers, index = tickers)
sig_attn_m = pd.DataFrame(data = np.nan, columns = tickers, index = tickers)
sig_sent_d = pd.DataFrame(data = np.nan, columns = tickers, index = tickers)
sig_sent_w = pd.DataFrame(data = np.nan, columns = tickers, index = tickers)
sig_sent_m = pd.DataFrame(data = np.nan, columns = tickers, index = tickers)

for i in range(len(sig_attn_d)):
    for j in range(i):
        print(i,j)
        sig_attn_d.iloc[i,j]= round(get_corrsig(corr_attn_d.iloc[i,j],"Daily"),2)
        sig_attn_w.iloc[i,j] = round(get_corrsig(corr_attn_w.iloc[i,j],"Weekly"),2)
        sig_attn_m.iloc[i,j] = round(get_corrsig(corr_attn_m.iloc[i,j],"Monthly"),2)
        sig_sent_d.iloc[i,j] = round(get_corrsig(corr_sent_d.iloc[i,j],"Daily"),2)
        sig_sent_w.iloc[i,j] = round(get_corrsig(corr_sent_w.iloc[i,j],"Weekly"),2)
        sig_sent_m.iloc[i,j] = round(get_corrsig(corr_sent_m.iloc[i,j],"Monthly"),2)


all_sig_sent = pd.DataFrame(columns=tickers, index=index_list)
all_sig_attn = pd.DataFrame(columns=tickers, index=index_list)

#Assembling all correlation significance
# Sentiment
all_sig_sent.iloc[0, :] = ["Daily"]*6
all_sig_sent.iloc[1:7, :] = sig_sent_d
all_sig_sent.iloc[7, :] = ["Weekly"]*6
all_sig_sent.iloc[8:14, :] = sig_sent_w
all_sig_sent.iloc[14, :] = ["Monthly"]*6
all_sig_sent.iloc[15:, :] = sig_sent_m

# Attention
all_sig_attn.iloc[0, :] = ["Daily"]*6
all_sig_attn.iloc[1:7, :] = sig_attn_d
all_sig_attn.iloc[7, :] = ["Weekly"]*6
all_sig_attn.iloc[8:14, :] = sig_attn_w
all_sig_attn.iloc[14, :] = ["Monthly"]*6
all_sig_attn.iloc[15:, :] = sig_attn_m

# Filling in values not existing
for i in range(len(all_sig_attn)):
    for j in range(len(tickers)):
        if pd.isna(all_sig_attn.iloc[i, j]):
            all_sig_attn.iloc[i, j] = ""
        if pd.isna(all_sig_sent.iloc[i, j]):
            all_sig_sent.iloc[i, j] = ""

#Saving to latex
with open(directory+'/LatexTables/sig_sent.tex', 'w') as tf:
    tf.write(all_sig_sent.to_latex())

with open(directory+'/LatexTables/sig_attn.tex', 'w') as tf:
    tf.write(all_sig_attn.to_latex())




# Correlations with returns
corrdf = pd.DataFrame(index=["Frequency"]+tickers,
                      columns=["SENT", "", "", "ATTN", "", ""])
corrdf.iloc[0, :] = frequencies*2
for i in range(len(tickers)):
    for j in range(len(frequencies)):
        at, se = get_corrs(frequencies[j], i)
        corrdf.iloc[1+i, j] = round(se, 3)
        corrdf.iloc[1+i, j+3] = round(at, 3)

with open(directory+'/LatexTables/corr_ret.tex', 'w') as tf:
    tf.write(corrdf.to_latex())




# Significance of correlations with return
corrsig = pd.DataFrame(index=["Frequency"]+tickers,
                       columns=["SENT", "", "", "ATTN", "", ""])
corrsig.iloc[0, :] = frequencies*2
for i in range(len(tickers)):
    for j in range(len(frequencies)*2):
        r = corrdf.iloc[i+1, j]
        freq = corrdf.iloc[0, j]
        corrsig.iloc[1+i, j] = round(get_corrsig(r, freq), 2)

# Saving it as a table
with open(directory+'/LatexTables/corr_sig.tex', 'w') as tf:
    tf.write(corrsig.to_latex())
