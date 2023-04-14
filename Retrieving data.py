import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import scipy.linalg as la
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from datetime import date
from pandas import ExcelWriter

##Dividing the data
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"
tickers = ["AMZN","AAPL","META","GOOGL","MSFT", "NFLX"]


#Reading in all the data
def read_raw(r,freq):
    file = directory+"/SavedData/allFilesRaw"+freq+".xlsx"
    raw = pd.read_excel(file,index_col=0,engine=None,sheet_name = tickers[r])
    return raw

def read_sta(r,freq):
    file = directory+"/SavedData/allFilesStandardized"+freq+".xlsx"
    sta = pd.read_excel(file,index_col=0,engine=None,sheet_name = tickers[r])
    return sta

def save_xls(list_dfs, labels, xls_path):
    with ExcelWriter(xls_path)as writer:
        for i in range(len(list_dfs)):
            list_dfs[i].to_excel(writer, sheet_name=labels[i])

#Listing all variables 
tvl = 'Trading Volume'
tpc = 'Twitter Publication Count'
tpsc = 'Twitter Positive Sentiment Count'
tnsc = 'Twitter Negative Sentiment Count'
npc = 'News Publication Count'
npsc = 'News Positive Sentiment Count'
nnsc = 'News Negative Sentiment Count'
si = 'SI'
sir = 'SI Ratio'
ivm = 'IV Midpoint'
pcr = 'P/C Ratio'
ti = 'AMZN'
sna = 'amazon stock'
svi = 'SVI'

#Sentiment
sentiment_list = [tpsc, tnsc, npsc, nnsc, ivm]
attention_list = [tvl, tpc, npc, svi]


def data_lag(data,list):
    dat = pd.DataFrame(index=data.iloc[1:,].index)
    colu = list
    for i in range(len(colu)):
        dat[list[i]] = np.array(data.iloc[1:,i])
        dat[list[i]+" 1 lag"] = np.array(data.iloc[:-1,i])
    return dat

def pca_split(data):
    cov = data.cov(numeric_only=False)
    corr = data.corr(numeric_only=False)
    eigval, eigvec = la.eig(cov)
    return cov, corr,  eigval.real, eigvec

def get_pca_split(data):
    data_att = data[attention_list]
    data_sent = data[sentiment_list]
    l_at = len(attention_list)
    l_se = len(sentiment_list)
    cols = attention_list+sentiment_list
    c1, c2 , e1, e2 = pca_split(data_att)
    d1, d2 , f1, f2 = pca_split(data_sent)

#Covariance matrix
    covar = pd.DataFrame(index = cols,columns = cols)
    covar.iloc[:l_at,:l_at] = c1 
    covar.iloc[-l_se:,-l_se:] = d1 
 
#Correlation Matrix
    corr = pd.DataFrame(index = cols,columns = cols)
    corr.iloc[:l_at,:l_at] = c2
    corr.iloc[-l_se:,-l_se:] = d2 

#Eigens
    eig_at =pd.DataFrame(index = ["EigenValues"]+attention_list+["Variance"],columns = attention_list)
    eig_at.iloc[0,:] = e1.real
    eig_at.iloc[1:-1,:] = e2
    eig_at = eig_at.sort_values(by=['EigenValues'],axis = 1,ascending = False)
    eig_at.iloc[-1,:] = eig_at.iloc[0,:]/eig_at.iloc[0,:].sum()
    eig_se =pd.DataFrame(index = ["EigenValues"]+sentiment_list+["Variance"],columns = sentiment_list)
    eig_se.iloc[0,:] = f1.real 
    eig_se.iloc[1:-1,:] = f2
    eig_se = eig_se.sort_values(by=['EigenValues'],axis = 1,ascending = False)
    eig_se.iloc[-1,:] = eig_se.iloc[0,:]/eig_se.iloc[0,:].sum()
    eigen = pd.DataFrame(index = ["EigenValues"]+cols+["Variance"],columns = cols)
    eigen.iloc[0,:l_at] = eig_at.iloc[0,:]
    eigen.iloc[0,-l_se:] = eig_se.iloc[0,:]
    eigen.iloc[1:l_at+1,:l_at] = eig_at.iloc[1:-1,:]
    eigen.iloc[-l_se-1:-1,l_at:] = eig_se.iloc[1:-1,:]
    eigen.iloc[-1,:l_at] = eig_at.iloc[-1,:]
    eigen.iloc[-1,l_at:] = eig_se.iloc[-1,:]
    
#Our variable ratios 
    ratios = []
    r1 = eig_at.iloc[1:-1,0]
    r2 = eig_se.iloc[1:-1,0]
    ratios.append(r1)
    ratios.append(r2)

    return covar, corr, eigen, ratios


#Getting the data 
frequency = "Weekly"
rawdata = read_raw(0,frequency)
data = read_sta(0,frequency)
ticker = tickers[0]
covar, corr, eigen, ratios = get_pca_split(data)

attention_ratio = np.matmul(data[attention_list],ratios[0])
sentiment_ratio = np.matmul(data[sentiment_list],ratios[1]) 

analysis = pd.DataFrame(rawdata['Price - Close - Monthly'])
analysis['Attention'] = attention_ratio
analysis['Sentiment'] = sentiment_ratio

stock_info = [rawdata, data,covar,corr, eigen]

rlab = ticker + " Raw data "+frequency
dlab = ticker + " Standardized data "+frequency
covlab = ticker + " covariance "+frequency
corrlab = ticker + " corrlation "+frequency
eiglab = ticker + " Eigens "+frequency

info_labels = [rlab, dlab, covlab, corrlab, eiglab]

save_xls(stock_info, info_labels, directory+"/SavedData/Amazon.xlsx")

test = pd.read_csv(directory+"/covtest.csv")
d = test.columns[0]
test = test.set_index(d)
