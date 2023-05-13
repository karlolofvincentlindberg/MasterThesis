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

# Function to read in our raw data


def read_data(r, freq):
    file = directory+"/18 CSVs - Final/"+tickers[r]+" - "+freq+".csv"
    data = pd.read_csv(file, index_col=0, engine=None)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data = data.set_index('Date')
    data['TR'] = data['TV']/data['SHOUT']
    return data


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


# Our variables
tvl = 'TV'
tpc = 'TC'
tpsc = 'TPC'
tnsc = 'TNC'
npc = 'NC'
npsc = 'NPC'
nnsc = 'NNC'
si = 'SI'
sir = 'SIR'
ivm = 'VMP'
pcr = 'PCR'
ti = 'SVI - T'
sna = 'SVI - S'
svi = 'SVI'
sho = 'SHOUT'

# Sentiment & attention lists
sentiment_list = [tpsc, tnsc, npsc, nnsc, pcr, ivm]
attention_list = [tvl, tpc, npc, svi]


mindate = dt.datetime(2015, 2, 1)
mindate = mindate.strftime('%Y-%m-%d')
maxdate = dt.datetime(2022, 1, 1)
maxdate = maxdate.strftime('%Y-%m-%d')

# Creating a list of all the raw variables we want to check
list_mix = ['Price']
for i in range(len(sentiment_list)):
    list_mix.append(sentiment_list[i])
for i in range(len(attention_list)):
    list_mix.append(attention_list[i])
list_mix.append('TR')

# Checking the stationarity for our raw varibles


def collinearity_matrix(cols, name):
    labels, column_list = create_lists(cols)
    dfmat = pd.DataFrame(data=None, index=labels, columns=column_list)
    for k in range(len(cols)):
        for j in range(len(frequencies)):
            frequency = frequencies[j]
            for i in range(len(tickers)):
                data = read_data(i, frequency)
                data = data.loc[:, list_mix]
                data = data[data.index >= mindate]
                data = data[data.index <= maxdate]
                data = data.diff()
                result = adfuller(data.iloc[1:, k])

                r = j*6+i
                c = 2*k
                if result[1] > 0.05:
                    dfmat.iloc[r, c] = "No"
                else:
                    dfmat.iloc[r, c] = "Yes"
    for k in range(len(cols)):
        for j in range(len(frequencies)):
            frequency = frequencies[j]
            for i in range(len(tickers)):
                data = read_data(i, frequency)
                data = data.loc[:, list_mix]
                data = data[data.index >= mindate]
                data = data[data.index <= maxdate]
                data = data.diff()
                kpss_stat, p_value, lags, critical_values = kpss(
                    data.iloc[1:, k])
                r = j*6+i
                c = 2*k+1
                if p_value < 0.05:
                    dfmat.iloc[r, c] = "No"
                else:
                    dfmat.iloc[r, c] = "Yes"
    dfmat.to_excel(f"{directory}/StationaritY/{name}.xlsx")
    return dfmat


df_mat = collinearity_matrix(list_mix, "Stationarity")

df_trans = df_mat.transpose()

df_daily = df_trans.iloc[:, :6]
df_daily.columns = tickers
with open(directory+'/LatexTables/Stationarities/raw_sta_d.tex', 'w') as tf:
    tf.write(df_daily.to_latex())

df_weekly = df_trans.iloc[:, 6:12]
df_weekly.columns = tickers
with open(directory+'/LatexTables/Stationarities/raw_sta_w.tex', 'w') as tf:
    tf.write(df_weekly.to_latex())

df_monthly = df_trans.iloc[:, 12:]
df_monthly.columns = tickers
with open(directory+'/LatexTables/Stationarities/raw_sta_m.tex', 'w') as tf:
    tf.write(df_monthly.to_latex())

# Checking stationarity for our computed ratios
# Reading in dataset to get columns and labels for results matrix
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/RatioData/"+ticker+".xlsx"
data = pd.read_excel(filename, index_col=0, engine=None,
                     sheet_name=ticker+" "+frequency)
cols = data.columns
labels, column_list = create_lists(cols)

# Creating dataframe to hold our results
dfmat = pd.DataFrame(data=None, index=labels, columns=column_list)

# Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe
for k in range(len(cols)):
    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            filename = directory+"/RatioDATA/"+ticker+".xlsx"
            data = pd.read_excel(filename, index_col=0,
                                 engine=None, sheet_name=ticker+" "+frequency)
            result = adfuller(data.iloc[1:, k])
            r = j*6+i
            c = 2*k
            if result[1] > 0.05:
                dfmat.iloc[r, c] = "No"
            else:
                dfmat.iloc[r, c] = "Yes"

# KPSS test - adding results to matrix and dataframe
for k in range(len(cols)):
    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            filename = directory+"/RatioDATA/"+ticker+".xlsx"
            data = pd.read_excel(filename, index_col=0,
                                 engine=None, sheet_name=ticker+" "+frequency)
            kpss_stat, p_value, lags, critical_values = kpss(data.iloc[1:, k])
            r = j*6+i
            c = 2*k+1
            if p_value < 0.05:
                dfmat.iloc[r, c] = "No"
            else:
                dfmat.iloc[r, c] = "Yes"
dfmat.to_excel(f"{directory}/Stationarity/ResultsStationarity.xlsx")

# Stationarity of computed ratios for linear regressions
# Daily
df_d = dfmat.transpose().iloc[:, :6]
df_d.columns = tickers
df_d = df_d.drop(['Log Return - ADF', 'Log Return - KPSS'])
with open(directory+'/LatexTables/Stationarities/Stationarity_d.tex', 'w') as tf:
    tf.write(df_d.to_latex())

# Weekly
df_w = dfmat.transpose().iloc[:, 6:12]
df_w.columns = tickers
df_w = df_w.drop(['Log Return - ADF', 'Log Return - KPSS'])
with open(directory+'/LatexTables/Stationarities/Stationarity_w.tex', 'w') as tf:
    tf.write(df_w.to_latex())

# Monthly
df_m = dfmat.transpose().iloc[:, 12:]
df_m.columns = tickers
df_m = df_m.drop(['Log Return - ADF', 'Log Return - KPSS'])
with open(directory+'/LatexTables/Stationarities/Stationarity_m.tex', 'w') as tf:
    tf.write(df_m.to_latex())

# Checking stationarity for our computed ratios - residualized data
# Reading in dataset to get columns and labels for results matrix
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/ResData/"+ticker+" "+frequency+".xlsx"
data = pd.read_excel(filename, index_col=0, engine=None)
colsss = data.columns
labels, column_list = create_lists(colsss)

# Creating dataframe to hold our results
dfmat_res = pd.DataFrame(data=None, index=labels, columns=column_list)

# Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe
for k in range(len(colsss)):
    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            filename = directory+"/ResData/"+ticker+" "+frequency+".xlsx"
            data = pd.read_excel(filename, index_col=0, engine=None)
            result = adfuller(data.iloc[1:, k])
            r = j*6+i
            c = 2*k
            if result[1] > 0.05:
                dfmat_res.iloc[r, c] = "No"
            else:
                dfmat_res.iloc[r, c] = "Yes"

# KPSS test - adding results to matrix and dataframe
for k in range(len(colsss)):
    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            filename = directory+"/ResData/"+ticker+" "+frequency+".xlsx"
            data = pd.read_excel(filename, index_col=0, engine=None)
            kpss_stat, p_value, lags, critical_values = kpss(data.iloc[1:, k])
            r = j*6+i
            c = 2*k+1
            if p_value < 0.05:
                dfmat_res.iloc[r, c] = "No"
            else:
                dfmat_res.iloc[r, c] = "Yes"
dfmat_res.to_excel(f"{directory}/Stationarity/ResResultsStationarity.xlsx")

# Stationarity of computed ratios for linear regressions
# Daily
df_d_res = dfmat_res.transpose().iloc[:, :6]
df_d_res.columns = tickers
df_d_res = df_d_res.drop(['Log Return - ADF', 'Log Return - KPSS'])
with open(directory+'/LatexTables/Stationarities/Stationarity_d_res.tex', 'w') as tf:
    tf.write(df_d_res.to_latex())

# Weekly
df_w_res = dfmat_res.transpose().iloc[:, 6:12]
df_w_res.columns = tickers
df_w_res = df_w_res.drop(['Log Return - ADF', 'Log Return - KPSS'])
with open(directory+'/LatexTables/Stationarities/Stationarity_w_res.tex', 'w') as tf:
    tf.write(df_w_res.to_latex())

# Monthly
df_m_res = dfmat_res.transpose().iloc[:, 12:]
df_m_res.columns = tickers
df_m_res = df_m_res.drop(['Log Return - ADF', 'Log Return - KPSS'])
with open(directory+'/LatexTables/Stationarities/Stationarity_m_res.tex', 'w') as tf:
    tf.write(df_m_res.to_latex())

# Checking stationarity for our computed ratios - orthogonalized data
# Reading in dataset to get columns and labels for results matrix
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/Orthogonalized/"+ticker+"Ratios.xlsx"
data = pd.read_excel(filename, index_col=0, engine=None)
colssss = data.columns
labels, column_list = create_lists(colssss)

# Creating dataframe to hold our results
dfmat_ort = pd.DataFrame(data=None, index=tickers, columns=column_list)

# Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe
for k in range(len(colsss)):
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/ResData/"+ticker+" "+frequency+".xlsx"
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
        filename = directory+"/ResData/"+ticker+" "+frequency+".xlsx"
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


# Exponential models
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/RatioData/"+ticker+".xlsx"
data = pd.read_excel(filename, index_col=0, engine=None,
                     sheet_name=ticker+" "+frequency)
colss = ['Log Return', 'ln(|ATTN|)', 'ln(|SENT|)', 'ATTN*SENT', 'ln(|deltaATTN|)', 'ln(|deltaSENT|)',
         'deltaATTN*deltaSENT', 'ln(|%deltaATTN|)', 'ln(|%deltaSENT|)', '%deltaATTN*%deltaSENT']
labelss, columns_list = create_lists(colss)


def transform_data(data):
    df = pd.DataFrame(index=data.index, columns=colss)
    for i in range(len(df)):
        df.iloc[i, 0] = data.iloc[i, 4]
        df.iloc[i, 1] = math.log(abs(data.iloc[i, 2]))
        df.iloc[i, 2] = math.log(abs(data.iloc[i, 3]))
        df.iloc[i, 3] = data.iloc[i, 3]*data.iloc[i, 2]
        df.iloc[i, 4] = math.log(abs(data.iloc[i, 5]))
        df.iloc[i, 5] = math.log(abs(data.iloc[i, 6]))
        df.iloc[i, 6] = data.iloc[i, 5]*data.iloc[i, 6]
        df.iloc[i, 7] = math.log(abs(data.iloc[i, 8]))
        df.iloc[i, 8] = math.log(abs(data.iloc[i, 9]))
        df.iloc[i, 9] = data.iloc[i, 8]*data.iloc[i, 9]
    return df


# Creating dataframe to hold our results
dfmat_log = pd.DataFrame(data=None, index=labelss, columns=columns_list)

# Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe
for k in range(len(colss)):
    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            filename = directory+"/RatioDATA/"+ticker+".xlsx"
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
            filename = directory+"/RatioDATA/"+ticker+".xlsx"
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
dfmat_log.to_excel(f"{directory}/Stationarity/LogResultsStationarity.xlsx")

# Stationarity of computed ratios for linear regressions
# Daily
df_d_log = dfmat_log.transpose().iloc[:, :6]
df_d_log.columns = tickers
with open(directory+'/LatexTables/Stationarities/Stationarity_d_log.tex', 'w') as tf:
    tf.write(df_d_log.to_latex())

# Weekly
df_w_log = dfmat_log.transpose().iloc[:, 6:12]
df_w_log.columns = tickers
with open(directory+'/LatexTables/Stationarities/Stationarity_w_log.tex', 'w') as tf:
    tf.write(df_w_log.to_latex())

# Monthly
df_m_log = dfmat_log.transpose().iloc[:, 12:]
df_m_log.columns = tickers
with open(directory+'/LatexTables/Stationarities/Stationarity_m_log.tex', 'w') as tf:
    tf.write(df_m_log.to_latex())


# Othogonalizing
dfmat_log_ort = pd.DataFrame(data=None, index=labelss, columns=columns_list)

# Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe
for k in range(len(colss)):
    for j in range(len(frequencies)):
        frequency = frequencies[j]
        for i in range(len(tickers)):
            ticker = tickers[i]
            filename = directory+"/Orthogonalized/"+ticker+"Ratios.xlsx"
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
            filename = directory+"/RatioDATA/"+ticker+".xlsx"
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
dfmat_log.to_excel(f"{directory}/Stationarity/LogResultsStationarity.xlsx")
