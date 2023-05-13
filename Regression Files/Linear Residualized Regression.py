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

#Linear 
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import linear_model

from datetime import date
from pandas import ExcelWriter

##Dividing the data
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"
tickers = ["AAPL","AMZN","GOOGL","META","MSFT", "NFLX"]

#Saving the file
def save_xls(list_dfs, labels, xls_path):
    with ExcelWriter(xls_path)as writer:
        for i in range(len(list_dfs)):
            list_dfs[i].to_excel(writer, sheet_name=labels[i])

#Regression model
def regression(y,x):
    x = sm.add_constant(x, prepend=True)
    model = sm.OLS(y,x)
    model_fit = model.fit()
    betas = model_fit.params
    t_stats = model_fit.tvalues
    r2 = model_fit.rsquared
    #print(model_fit.summary())
    return betas, t_stats, r2

#First linear regression
##R_t = B*S_t-1% + B*A_t-1
lin_one = "R_t = B*S_t-1% + B*dA_t-1"
def regression_one(df, lag):
    x_attn = np.array(df.iloc[1:-lag,1]).reshape((-1,1))
    x_sent = np.array(df.iloc[1:-lag,2]).reshape((-1,1))
    y = np.array(df.iloc[lag+1:,-3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y,x)
    return beta, t_stat, rsq

#Second Linear Regression
##R_t = B*deltaS_t-1 + B*deltaA_t-1
lin_two = "R_t = B*deltaS_t-1 + B*deltaA_t-1"
def regression_two(df, lag):
    x_attn = np.array(df.iloc[1:-lag,-5]).reshape((-1,1))
    x_sent = np.array(df.iloc[1:-lag,-4]).reshape((-1,1))
    y = np.array(df.iloc[lag+1:,-3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y,x)
    return beta, t_stat, rsq

#Third linear regression
##R_t = B*%deltaS_t-1% + B*%deltaA_t-1
lin_three = "R_t = B*%deltaS_t-1% + B*%deltaA_t-1"
def regression_three(df, lag):
    x_attn = np.array(df.iloc[1:-lag,-2]).reshape((-1,1))
    x_sent = np.array(df.iloc[1:-lag,-1]).reshape((-1,1))
    y = np.array(df.iloc[lag+1:,-3])
    x = np.hstack((x_attn, x_sent))
    beta, t_stat, rsq = regression(y,x)
    return beta, t_stat, rsq

#Transforming dataset and creating residuals
def data_res(df):
    data = pd.DataFrame(df)
    data['Price lag'] = np.nan
    for i in range(1,len(data)):
        data.iloc[i,-1] = data.iloc[i-1,0]
    bs, ts, r2 = regression(data.iloc[1:,0],data.iloc[1:,-1])
    alpha = bs[0]
    beta = bs[1]
    data['Res'] = data['Price'] - alpha - beta*data['Price lag']
    data['Return'] = data['Res']
    data = data.drop(['Price lag','Res'], axis = 1)
    data = data.rename(columns={'Return':'Residual'})
    return data

#Listing our frequencies & intervals
frequencies = ["Daily","Weekly","Monthly"]
intervals = ["Daily","Daily","Weekly","Weekly","Monthly","Monthly"]

#Creating our index list
index_list = ["Proxy"]
for i in range(len(tickers)):
     index_list.append(tickers[i])
     index_list.append(" ")


#Residual daily regression
res_d_reg = pd.DataFrame(index = index_list, columns = ["Model 1","","Model 2","","Model 3",""] )

#Residual weekly regression
res_w_reg = pd.DataFrame(index = index_list, columns = ["Model 1","","Model 2","","Model 3",""] )

#Residual monthly regression
res_m_reg = pd.DataFrame(index = index_list, columns = ["Model 1","","Model 2","","Model 3",""] )


#Regressing using residuals
# P_t = a + B*P_t-1 +res
# res = p_t - a - b*P_t-1 
dfs = []
tickers_lab = []

#Residuals daily 
res_d_reg.iloc[0,:] = ['ATTN','SENT','ATTN','SENT','ATTN','SENT']
for i in range(len(tickers)):
    frequency = "Daily"
    ticker = tickers[i]
    filename = directory+"/SavedData/RatioDATA/"+ticker+".xlsx"
    df = pd.read_excel(filename,index_col=0,engine=None,sheet_name = ticker+" "+frequency)
    data = data_res(df)
    dfs.append(data)
    data = data.drop('Price', axis=1)
    print(i)
    print(data.head())
    tickers_lab.append(f"{ticker} {frequency}")
    b1, t_sta1, r1 = regression_one(data,1)
    b2, t_sta2, r2 = regression_two(data,1)
    b3, t_sta3, r3 = regression_three(data,1)
    res_d_reg.iloc[2*i+1,0] = f"{round(b1[1]*10**3,2)}"
    res_d_reg.iloc[2*i+2,0] = f"({round(t_sta1[1],2)})"
    res_d_reg.iloc[2*i+1,1] = f"{round(b1[2]*10**3,2)}"
    res_d_reg.iloc[2*i+2,1] = f"({round(t_sta1[2],2)})"
    res_d_reg.iloc[2*i+1,2] = f"{round(b2[1]*10**3,2)}"
    res_d_reg.iloc[2*i+2,2] = f"({round(t_sta2[1],2)})"
    res_d_reg.iloc[2*i+1,3] = f"{round(b2[2]*10**3,2)}"
    res_d_reg.iloc[2*i+2,3] = f"({round(t_sta2[2],2)})"
    res_d_reg.iloc[2*i+1,4] = f"{round(b3[1]*10**3,2)}"
    res_d_reg.iloc[2*i+2,4] = f"({round(t_sta3[1],2)})"
    res_d_reg.iloc[2*i+1,5] = f"{round(b3[2]*10**3,2)}"
    res_d_reg.iloc[2*i+2,5] = f"({round(t_sta3[2],2)})"

#Residuals weekly 
res_w_reg.iloc[0,:] = ['ATTN','SENT','ATTN','SENT','ATTN','SENT']
for i in range(len(tickers)):
    frequency = "Weekly"
    ticker = tickers[i]
    filename = directory+"/SavedData/RatioDATA/"+ticker+".xlsx"
    df = pd.read_excel(filename,index_col=0,engine=None,sheet_name = ticker+" "+frequency)
    data = data_res(df)
    dfs.append(data)
    data = data.drop('Price', axis=1)
    print(i)
    print(data.head())
    tickers_lab.append(f"{ticker} {frequency}")
    b1, t_sta1, r1 = regression_one(data,1)
    b2, t_sta2, r2 = regression_two(data,1)
    b3, t_sta3, r3 = regression_three(data,1)
    res_w_reg.iloc[2*i+1,0] = f"{round(b1[1]*10**3,2)}"
    res_w_reg.iloc[2*i+2,0] = f"({round(t_sta1[1],2)})"
    res_w_reg.iloc[2*i+1,1] = f"{round(b1[2]*10**3,2)}"
    res_w_reg.iloc[2*i+2,1] = f"({round(t_sta1[2],2)})"
    res_w_reg.iloc[2*i+1,2] = f"{round(b2[1]*10**3,2)}"
    res_w_reg.iloc[2*i+2,2] = f"({round(t_sta2[1],2)})"
    res_w_reg.iloc[2*i+1,3] = f"{round(b2[2]*10**3,2)}"
    res_w_reg.iloc[2*i+2,3] = f"({round(t_sta2[2],2)})"
    res_w_reg.iloc[2*i+1,4] = f"{round(b3[1]*10**3,2)}"
    res_w_reg.iloc[2*i+2,4] = f"({round(t_sta3[1],2)})"
    res_w_reg.iloc[2*i+1,5] = f"{round(b3[2]*10**3,2)}"
    res_w_reg.iloc[2*i+2,5] = f"({round(t_sta3[2],2)})"

#Residuals monthly 
res_m_reg.iloc[0,:] = ['ATTN','SENT','ATTN','SENT','ATTN','SENT']
for i in range(len(tickers)):
    frequency = "Monthly"
    ticker = tickers[i]
    filename = directory+"/SavedData/RatioDATA/"+ticker+".xlsx"
    df = pd.read_excel(filename,index_col=0,engine=None,sheet_name = ticker+" "+frequency)
    data = data_res(df)
    dfs.append(data)
    data = data.drop('Price', axis=1)
    print(i)
    print(data.head())
    tickers_lab.append(f"{ticker} {frequency}")
    b1, t_sta1, r1 = regression_one(data,1)
    b2, t_sta2, r2 = regression_two(data,1)
    b3, t_sta3, r3 = regression_three(data,1)
    res_m_reg.iloc[2*i+1,0] = f"{round(b1[1]*10**3,2)}"
    res_m_reg.iloc[2*i+2,0] = f"({round(t_sta1[1],2)})"
    res_m_reg.iloc[2*i+1,1] = f"{round(b1[2]*10**3,2)}"
    res_m_reg.iloc[2*i+2,1] = f"({round(t_sta1[2],2)})"
    res_m_reg.iloc[2*i+1,2] = f"{round(b2[1]*10**3,2)}"
    res_m_reg.iloc[2*i+2,2] = f"({round(t_sta2[1],2)})"
    res_m_reg.iloc[2*i+1,3] = f"{round(b2[2]*10**3,2)}"
    res_m_reg.iloc[2*i+2,3] = f"({round(t_sta2[2],2)})"
    res_m_reg.iloc[2*i+1,4] = f"{round(b3[1]*10**3,2)}"
    res_m_reg.iloc[2*i+2,4] = f"({round(t_sta3[1],2)})"
    res_m_reg.iloc[2*i+1,5] = f"{round(b3[2]*10**3,2)}"
    res_m_reg.iloc[2*i+2,5] = f"({round(t_sta3[2],2)})"

#Saving our residuals regression table
#Daily
with open(directory+'/LatexTables/Regressions/Residual/resid_d.tex','w') as tf:
    tf.write(res_d_reg.to_latex())  

#Weekly
with open(directory+'/LatexTables/Regressions/Residual/resid_w.tex','w') as tf:
    tf.write(res_w_reg.to_latex())  

#Monthly
with open(directory+'/LatexTables/Regressions/Residual/resid_m.tex','w') as tf:
    tf.write(res_m_reg.to_latex())  

#Saving our residual data
for i in range(len(dfs)):
    filepath = directory+"/SavedData/ResData/"+tickers_lab[i]+".xlsx"
    dfs[i].to_excel(filepath)

    