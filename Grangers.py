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
from statsmodels.tsa.stattools import grangercausalitytests 

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

def get_data_se(freq, i):
    df = pd.DataFrame()
    df['SENT'] = get_sent(freq,i)
    df['Return'] = get_ret(freq, i)
    return df 

def get_data(freq, i):
    df = pd.DataFrame()
    df['Return'] = get_ret(freq, i)
    df['ATTN'] = get_attn(freq,i)
    df['SENT'] = get_sent(freq,i)
    return df.iloc[1:,:]

def hermione_granger(i,r,c):
    data = get_data("Daily", i)
    data_trans = pd.DataFrame([data.iloc[:,r],data.iloc[:,c]]).transpose()
    results = grangercausalitytests(data_trans, maxlag=1)
    print(data_trans.head())
    # print the results
    for lag in results.keys():
        print('Lag:', lag)
        print('F-statistic:', results[lag][0]['params_ftest'][0])
        print('p-value:', results[lag][0]['params_ftest'][1])
        print(' ')
    p_val = results[lag][0]['params_ftest'][1]
    return p_val

cols = ['Return(X)', "ATTN(X)", "SENT(X)"]  
ind = ['Return(Y)', "ATTN(Y)", "SENT(Y)"]  
matrices = []

for i in range(6):
    print(tickers[i])
    matrix = pd.DataFrame(columns = cols, index = ind)
    for r in range(3):
        for c in range(3):
            p = hermione_granger(i,r,c)
            matrix.iloc[r,c] = round(p,4)
    matrices.append(matrix)
    print(matrix)

hermione_grangers = pd.DataFrame(columns=cols, index =([""]+ind)*6)

for i in range(6):
    print(tickers[i])
    print(matrices[i])
    hermione_grangers.iloc[i*4,:] = tickers[i]
    hermione_grangers.iloc[i*4+1:i*4+4,:] = matrices[i]

with open(directory+'/LatexTables/Grangers.tex', 'w') as tf:
    tf.write(hermione_grangers.to_latex())
