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

#Listing our frequencies & intervals
frequencies = ["Daily","Weekly","Monthly"]
intervals = ["Daily","Daily","Weekly","Weekly","Monthly","Monthly"]

#Creating our index list
index_list = ["Proxy"]
for i in range(len(tickers)):
     index_list.append(tickers[i])
     index_list.append(" ")


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

#Creating x variables when using them in exponential models
def create_xvars(x1,x2):
    xa = []
    xb = x2
    for i in range(len(x1)):
        x = math.log(abs(x1[i]))
        xa.append(x)
    x_all = np.hstack((np.array(xa).reshape((-1,1)), xb))
    return x_all 


### Introducing our exponential models 
#First exponential model 
#Option 1
ex_one_one = "log R_t = B*ln(|A_t-1|) + B*S_t-1"
#Option 2
ex_one_two = "log R_t = B*ln(|S_t-1|) + B*A_t-1"
#Option 3
ex_one_three = "log R_t = ln(C) + B*A_t-1*S_t-1"
def expo_one(df,lag):
    x_attn = np.array(df.iloc[1:-lag,1]).reshape((-1,1))
    x_sent = np.array(df.iloc[1:-lag,2]).reshape((-1,1))
    y = np.array(df.iloc[lag+1:,3])
    x1 = create_xvars(x_attn,x_sent)
    x2 = create_xvars(x_sent,x_attn)
    x3 = x_attn*x_sent
    b1 , t_s1, rsq1 = regression(y,x1)
    b2 , t_s2, rsq2 = regression(y,x2)
    b3 , t_s3, rsq3 = regression(y,x3)
    return b1, b2, b3, t_s1,t_s2,t_s3

#First exponential model 
#Option 1
ex_two_one = "log R_t = B*ln(|deltaA_t-1|) + B*deltaS_t-1"
#Option 2
ex_two_two = "log R_t = B*ln(|deltaS_t-1|) + B*deltaA_t-1"
#Option 3
ex_two_three = "log R_t = ln(C) + B*deltaA_t-1*deltaS_t-1"
def expo_two(df,lag):
    x_attn = np.array(df.iloc[1:-lag,-5]).reshape((-1,1))
    x_sent = np.array(df.iloc[1:-lag,-4]).reshape((-1,1))
    y = np.array(df.iloc[lag+1:,3])
    x1 = create_xvars(x_attn,x_sent)
    x2 = create_xvars(x_sent,x_attn)
    x3 = x_attn*x_sent
    b1 , t_s1, rsq1 = regression(y,x1)
    b2 , t_s2, rsq2 = regression(y,x2)
    b3 , t_s3, rsq3 = regression(y,x3)
    return b1, b2, b3, t_s1,t_s2,t_s3

#Third exponential model 
#Option 1
ex_three_one = "log R_t = B*ln(|%deltaA_t-1|) + B*%deltaS_t-1"
#Option 2
ex_three_two = "log R_t = B*ln(|%deltaS_t-1|) + B*%deltaA_t-1"
#Option 3
ex_three_three = "log R_t = ln(C) + B*%deltaA_t-1*%deltaS_t-1"
def expo_three(df,lag):
    x_attn = np.array(df.iloc[1:-lag,-2]).reshape((-1,1))
    x_sent = np.array(df.iloc[1:-lag,-1]).reshape((-1,1))
    y = np.array(df.iloc[lag+1:,3])
    x1 = create_xvars(x_attn,x_sent)
    x2 = create_xvars(x_sent,x_attn)
    x3 = x_attn*x_sent
    b1 , t_s1, rsq1 = regression(y,x1)
    b2 , t_s2, rsq2 = regression(y,x2)
    b3 , t_s3, rsq3 = regression(y,x3)
    return b1, b2, b3, t_s1,t_s2,t_s3

#First exponential models 
df_em1_1 = pd.DataFrame(index = index_list, columns = intervals )
df_em1_2 = pd.DataFrame(index = index_list, columns = intervals )
df_em1_3 = pd.DataFrame(index = index_list, columns = intervals )

#Second exponential models 
df_em2_1 = pd.DataFrame(index = index_list, columns = intervals )
df_em2_2 = pd.DataFrame(index = index_list, columns = intervals )
df_em2_3 = pd.DataFrame(index = index_list, columns = intervals )

#Third exponential models 
df_em3_1 = pd.DataFrame(index = index_list, columns = intervals )
df_em3_2 = pd.DataFrame(index = index_list, columns = intervals )
df_em3_3 = pd.DataFrame(index = index_list, columns = intervals )

#Main exponential model 1

frequency = "Monthly"
df_em1_1.iloc[0, :] = ["ln(ATTN)", "SENT", "ln(ATTN)",
                       "SENT", "ln(ATTN)", "SENT"]
df_em1_2.iloc[0, :] = [
    "ATTN", "ln(SENT)", "ATTN", "ln(SENT)", "ATTN", "ln(SENT)"]
df_em1_3.iloc[0, :] = ["ln(C)", "ATTN*SENT",
                       "ln(C)", "ATTN*SENT", "ln(C)", "ATTN*SENT"]
for i in range(len(tickers)):
    ticker = tickers[i]
    filename = directory+"/SavedData/RatioDATA/"+ticker+".xlsx"
    data = pd.read_excel(filename, index_col=0, engine=None)
    data = data.drop('Price', axis=1)
    b1, b2, b3, t_s1, t_s2, t_s3 = expo_one(data, 1)
    print(b3)
    df_em1_1.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    df_em1_1.iloc[2*i+1, 1] = f"{round(b1[2]*10**3,2)}"
    df_em1_1.iloc[2*i+2, 0] = f"({round(t_s1[1],2)})"
    df_em1_1.iloc[2*i+2, 1] = f"({round(t_s1[2],2)})"
    df_em1_2.iloc[2*i+1, 1] = f"{round(b2[1]*10**3,2)}"
    df_em1_2.iloc[2*i+1, 0] = f"{round(b2[2]*10**3,2)}"
    df_em1_2.iloc[2*i+2, 1] = f"({round(t_s2[1],2)})"
    df_em1_2.iloc[2*i+2, 0] = f"({round(t_s2[2],2)})"
    df_em1_3.iloc[2*i+1, 1] = f"{round(b3[1]*10**3,2)}"
    df_em1_3.iloc[2*i+1, 0] = f"{round(b3[0]*10**3,2)}"
    df_em1_3.iloc[2*i+2, 1] = f"({round(t_s3[1],2)})"
    df_em1_3.iloc[2*i+2, 0] = f"({round(t_s3[0],2)})"

#Main exponential model 2
frequency = "Monthly"
df_em2_1.iloc[0, :] = [
    "ln(ATTN)", "SENT", "ln(ATTN)", "SENT", "ln(ATTN)", "SENT"]
df_em2_2.iloc[0, :] = [
    "ATTN", "ln(SENT)", "ATTN", "ln(SENT)", "ATTN", "ln(SENT)"]
df_em2_3.iloc[0, :] = ["ln(C)", "ATTN*SENT",
                       "ln(C)", "ATTN*SENT", "ln(C)", "ATTN*SENT"]
for i in range(len(tickers)):
    ticker = tickers[i]
    filename = directory+"/SavedData/RatioDATA/"+ticker+".xlsx"
    data = pd.read_excel(filename, index_col=0, engine=None)
    data = data.drop('Price', axis=1)
    b1, b2, b3, t_s1, t_s2, t_s3 = expo_two(data, 1)
    print(b3)
    df_em2_1.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    df_em2_1.iloc[2*i+1, 1] = f"{round(b1[2]*10**3,2)}"
    df_em2_1.iloc[2*i+2, 0] = f"({round(t_s1[1],2)})"
    df_em2_1.iloc[2*i+2, 1] = f"({round(t_s1[2],2)})"
    df_em2_2.iloc[2*i+1, 1] = f"{round(b2[1]*10**3,2)}"
    df_em2_2.iloc[2*i+1, 0] = f"{round(b2[2]*10**3,2)}"
    df_em2_2.iloc[2*i+2, 1] = f"({round(t_s2[1],2)})"
    df_em2_2.iloc[2*i+2, 0] = f"({round(t_s2[2],2)})"
    df_em2_3.iloc[2*i+1, 1] = f"{round(b3[1]*10**3,2)}"
    df_em2_3.iloc[2*i+1, 0] = f"{round(b3[0]*10**3,2)}"
    df_em2_3.iloc[2*i+2, 1] = f"({round(t_s3[1],2)})"
    df_em2_3.iloc[2*i+2, 0] = f"({round(t_s3[0],2)})"

#Main exponential model 3

frequency = "Monthly"
df_em3_1.iloc[0,:] = ["ln(|ATTN|)","SENT","ln(|ATTN|)","SENT","ln(|ATTN|)","SENT"]
df_em3_2.iloc[0,:] = ["ATTN","ln(|SENT|)","ATTN","ln(|SENT|)","ATTN","ln(|SENT|)"]
df_em3_3.iloc[0,:] = ["ln(C)","ATTN*SENT","ln(C)","ATTN*SENT","ln(C)","ATTN*SENT"]
for i in range(len(tickers)):
    ticker = tickers[i]
    filename = directory+"/SavedData/RatioDATA/"+ticker+".xlsx"
    data = pd.read_excel(filename,index_col=0,engine=None)
    data = data.drop('Price', axis=1)
    b1, b2, b3, t_s1, t_s2, t_s3 = expo_three(data, 1)
    print(b3)
    df_em3_1.iloc[2*i+1, 0] = f"{round(b1[1]*10**3,2)}"
    df_em3_1.iloc[2*i+1, 1] = f"{round(b1[2]*10**3,2)}"
    df_em3_1.iloc[2*i+2, 0] = f"({round(t_s1[1],2)})"
    df_em3_1.iloc[2*i+2, 1] = f"({round(t_s1[2])})"
    df_em3_2.iloc[2*i+1, 1] = f"{round(b2[1]*10**3,2)}"
    df_em3_2.iloc[2*i+1, 0] = f"{round(b2[2]*10**3,2)}"
    df_em3_2.iloc[2*i+2, 1] = f"({round(t_s2[1],2)})"
    df_em3_2.iloc[2*i+2, 0] = f"({round(t_s2[2],2)})"
    df_em3_3.iloc[2*i+1, 1] = f"{round(b3[1]*10**3,2)}"
    df_em3_3.iloc[2*i+1, 0] = f"{round(b3[0]*10**3,2)}"
    df_em3_3.iloc[2*i+2, 1] = f"({round(t_s3[1],2)})"
    df_em3_3.iloc[2*i+2, 0] = f"({round(t_s3[0],2)})"

#Gathering all exponential models in one dataframe
dataframes = [df_em1_1, df_em1_2, df_em1_3, df_em2_1, df_em2_2, df_em2_3, df_em3_1,df_em3_2, df_em3_3]
datalabels = ["em1_1_ort","em1_2_ort","em1_3_ort","em2_1_ort","em2_2_ort","em2_3_ort","em3_1_ort","em3_2_ort","em3_3_ort"]

models1 = ["Model 5.1","", "Model 5.2", "","Model 5.3","","Model 5.4",""]
models2 = ["Model 5.5","", "Model 5.6", "","Model 5.7","","Model 5.8",""]

df_join_one = pd.DataFrame(index = df_em1_1.index, columns=models1)

df_join_two = pd.DataFrame(index = df_em1_1.index, columns=models2)

df_join_one.iloc[:,:2] = df_em1_1.iloc[:,0:2]
df_join_one.iloc[:,2:4] = df_em1_2.iloc[:,0:2]
df_join_one.iloc[:,4:6] = df_em1_3.iloc[:,0:2]
df_join_one.iloc[:,6:8] = df_em2_1.iloc[:,0:2]

df_join_two.iloc[:,:2] = df_em2_3.iloc[:,0:2]
df_join_two.iloc[:,2:4] = df_em3_1.iloc[:,0:2]
df_join_two.iloc[:,4:6] = df_em3_2.iloc[:,0:2]
df_join_two.iloc[:,6:8] = df_em3_3.iloc[:,0:2]


with open(directory+'/LatexTables/Regressions/Exponential/Ortho1.tex','w') as tf:
    tf.write(df_join_one.to_latex())  

with open(directory+'/LatexTables/Regressions/Exponential/Ortho2.tex','w') as tf:
    tf.write(df_join_two.to_latex())  
