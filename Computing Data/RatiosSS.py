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


# Saving the file
def save_xls(list_dfs, labels, xls_path):
    with ExcelWriter(xls_path)as writer:
        for i in range(len(list_dfs)):
            list_dfs[i].to_excel(writer, sheet_name=labels[i])


proxies = ['Company', 'Obs.', 'Mean', 'StdDev.',
           'Min', 'Max', 'Skewness', 'Kurtosis']
index_list = ['Daily']*6+['Weekly']*6+['Monthly']*6

attn_df = pd.DataFrame(columns=proxies, index=index_list)
sent_df = pd.DataFrame(columns=proxies, index=index_list)

for j in range(len(frequencies)):
    frequency = frequencies[j]
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        column = data['ATTN']
        obs = column.count()
        mean = column.mean()
        std = column.std()
        max = column.max()
        min = column.min()
        skew = column.skew()
        kurt = column.kurtosis()
        attn_df.iloc[j*6+i, 0] = ticker
        attn_df.iloc[j*6+i, 1] = obs
        attn_df.iloc[j*6+i, 2] = round(mean*100, 4)
        attn_df.iloc[j*6+i, 3] = round(std, 4)
        attn_df.iloc[j*6+i, 4] = round(max, 4)
        attn_df.iloc[j*6+i, 5] = round(min, 4)
        attn_df.iloc[j*6+i, 6] = round(skew, 4)
        attn_df.iloc[j*6+i, 7] = round(kurt, 4)

for j in range(len(frequencies)):
    frequency = frequencies[j]
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/RatioDATA/"+ticker+".xlsx"
        data = pd.read_excel(filename, index_col=0,
                             engine=None, sheet_name=ticker+" "+frequency)
        column = data['SENT']
        obs = column.count()
        mean = column.mean()
        std = column.std()
        max = column.max()
        min = column.min()
        skew = column.skew()
        kurt = column.kurtosis()
        sent_df.iloc[j*6+i, 0] = ticker
        sent_df.iloc[j*6+i, 1] = obs
        sent_df.iloc[j*6+i, 2] = round(mean*100, 4)
        sent_df.iloc[j*6+i, 3] = std
        sent_df.iloc[j*6+i, 4] = round(max, 4)
        sent_df.iloc[j*6+i, 5] = round(min, 4)
        sent_df.iloc[j*6+i, 6] = round(skew, 4)
        sent_df.iloc[j*6+i, 7] = round(kurt, 4)


# Saving our tables to latex
with open(directory+'/LatexTables/attnss.tex', 'w') as tf:
    tf.write(attn_df.to_latex())

with open(directory+'/LatexTables/sentss.tex', 'w') as tf:
    tf.write(sent_df.to_latex())
