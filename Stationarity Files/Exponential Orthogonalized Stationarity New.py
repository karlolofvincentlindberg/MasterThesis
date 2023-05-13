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

#Linear 
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import linear_model

from datetime import date
from pandas import ExcelWriter

##Dividing the data
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"
tickers = ["AMZN","AAPL","META","GOOGL","MSFT", "NFLX"]

#Our frequencies 
frequencies = ["Daily","Weekly","Monthly"]

#Functions to create lists from columns of dataset 
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


###Exponential models 
ticker = tickers[0]
frequency = "Monthly"
filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx" 
data = pd.read_excel(filename,index_col=0,engine=None)
colss = ['Log Return','ln(ATTN)','ln(deltaATTN)','ln(%deltaATTN)']
labelss, columns_list = create_lists(colss)



def transform_data(data):
    df = pd.DataFrame(index  = data.index, columns = colss)
    for i in range(1,len(df)):
        df.iloc[i,0] = data.iloc[i,0]
        df.iloc[i,1] = math.log(data.iloc[i,1]-data.iloc[:,1].min()+1)
        df.iloc[i,2] = math.log(data.iloc[i,2]-data.iloc[:,2].min()+1)
        df.iloc[i,3] = math.log(data.iloc[i,3]-data.iloc[:,3].min()+1)
    return df


#Creating dataframe to hold our results 
dfmat_log = pd.DataFrame(data = None, index=tickers,columns=columns_list)

#Adjusted Dickey Fuller Test - adding resutls to matrix and dataframe
for k in range(len(colss)):
    frequency = "Monthly"
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
        data = pd.read_excel(filename,index_col=0,engine=None)
        new_data = transform_data(data[['Log Return','ATTN','Delta ATTN','%Delta ATTN']])
        result = adfuller(new_data.iloc[1:,k])
        r = i
        c = 2*k
        if result[1] >0.05:
            dfmat_log.iloc[r,c] = "No"
        else:
            dfmat_log.iloc[r,c] = "Yes"

#KPSS test - adding results to matrix and dataframe
for k in range(len(colss)):
    frequency = "Monthly"
    for i in range(len(tickers)):
        ticker = tickers[i]
        filename = directory+"/SavedData/Orthogonalized/"+ticker+"Ratios.xlsx"
        data = pd.read_excel(filename,index_col=0,engine=None)
        new_data = transform_data(data[['Log Return','ATTN','Delta ATTN','%Delta ATTN']])
        kpss_stat, p_value, lags, critical_values = kpss(new_data.iloc[1:,k])
        r = i
        c = 2*k+1
        if p_value<0.05:
            dfmat_log.iloc[r,c] = "No"
        else:
            dfmat_log.iloc[r,c] = "Yes"
dfmat_log.to_excel(f"{directory}/Stationarity/newMDLogResultsStationarity.xlsx")

#Stationarity of computed ratios for linear regressions
#Daily
df_log = dfmat_log.transpose()
with open(directory+'/LatexTables/Stationarities/Stationarity_lognew_md.tex','w') as tf:
    tf.write(df_log.to_latex())

    