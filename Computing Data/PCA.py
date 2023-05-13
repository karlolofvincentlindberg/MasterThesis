import pandas as pd
import numpy as np
import seaborn as sns
import math as math
import matplotlib.pyplot as plt
import datetime as dt
import scipy.linalg as la
import scipy.optimize as optimize
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import linear_model

from datetime import date
from pandas import ExcelWriter

# Dividing the data
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"
tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX"]

# Reading the data


def read_data(r, freq):
    file = directory+"/18 CSVs - Final/"+tickers[r]+" - "+freq+".csv"
    data = pd.read_csv(file, index_col=0, engine=None)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data = data.set_index('Date')
    return data

# Reading in all the data


def read_raw(r, freq):
    file = directory+"/SavedData/allFilesRaw"+freq+".xlsx"
    raw = pd.read_excel(file, index_col=0, engine=None, sheet_name=tickers[r])
    return raw

diff_datas_attn = []
diff_datas_sent = []

# Reading standardized data
sta_datas_att =[]
sta_datas_sent =[]
def read_sta(r, freq):
    file = directory+"/SavedData/allFilesStandardized"+freq+".xlsx"
    sta = pd.read_excel(file, index_col=0, engine=None, sheet_name=tickers[r])
    return sta

# Saving the file


def save_xls(list_dfs, labels, xls_path):
    with ExcelWriter(xls_path)as writer:
        for i in range(len(list_dfs)):
            list_dfs[i].to_excel(writer, sheet_name=labels[i])

# Standardizing


def standardize(df):
    dfs = (df-df.mean())/(df.std())
    return dfs

# Getting the lags


def data_lag(data, list):
    dat = pd.DataFrame(index=data.iloc[1:,].index)
    colu = list
    for i in range(len(colu)):
        dat[list[i]] = np.array(data.iloc[1:, i])
        dat[list[i]+" 1 lag"] = np.array(data.iloc[:-1, i])
    return dat

# Function to get the index position of a given value


def get_index(array, value):
    for i in range(len(array)):
        if array[i] == value:
            break
    return i

# Function to get the principal component of given dataset


def get_pc_one(data):
    pca = PCA(n_components=len(data.columns)).fit(data)
    var = pca.explained_variance_[0]
    pca_one = pca.components_[0]
    '''
#Enforcing kaizer rule
    for i in range(1,len(pca.components_)):
        if pca.explained_variance_[i]>1:
            var+= pca.explained_variance_[i]
            pca_one+= pca.components_[i]
    '''
    # print(var/len(pca.explained_variance_))
    ratio = np.matmul(data, pca_one.T)
    return np.array(ratio), var, pca_one

# Get the correlation of each variable and lag to the created ratio


def get_corrs(data):
    corrs = []
    for i in range(len(data.iloc[0, :])-1):
        cr = data.iloc[:, i].corr(data.iloc[:, -1])
        corrs.append(cr)
    return corrs

# Function to find the column of each pair with the highest absolute value of correlation


def get_columns(array):
    col_ind = []
    for i in range(int(len(array)/2)):
        one = abs(array[2*i])
        two = abs(array[2*i+1])
        if one >= two:
            col_ind.append(2*i)
        else:
            col_ind.append(2*i+1)
    return col_ind

# Retrieving the chosen columns from the dataset


def get_cols(data):
    corrs = get_corrs(data)
    col_inds = get_columns(corrs)
    cols = data.columns
    colss = []
    for i in range(len(col_inds)):
        ci = col_inds[i]
        col = cols[ci]
        colss.append(col)
    return colss, corrs

# Rebalancing our ratio to get a standard deviation of 1


def optimizing(data, pca, ratio):
    def correlation(x):
        new_ratio = np.matmul(data, x.T)
        df = pd.DataFrame(index=new_ratio.index)
        df["New Ratio"] = new_ratio
        df['Old Ratio'] = ratio
        corr = df.iloc[:, -1].corr(df.iloc[:, 0])
        return corr*-1
    cons = ({'type': 'eq', 'fun': lambda x: np.matmul(data, x).std()-1})
    res = optimize.minimize(fun=correlation, x0=pca, method='SLSQP', options={
                            'ftol': 1e-9}, constraints=cons)
    new_pca = res.x
    result = np.matmul(data, new_pca)
    return np.array(result), new_pca

# Getting our ratio


def get_ratio(data, list):
    dat = data_lag(data, list)
    dat['Ratio'], var_one, pca_one = get_pc_one(dat)
    datta = pd.DataFrame(dat)
    cols, corrs = get_cols(dat)
    dat = dat[cols]
    dat_ratio, var_two, pca_two = get_pc_one(dat)
    new_ratio, pca_opt = optimizing(dat, pca_two, dat_ratio)
    dat['NewRatio'] = np.nan
    for i in range(len(dat)):
        dat.iloc[i, -1] = new_ratio[i]
    cor = dat['NewRatio'].corr(datta['Ratio'])
    pca = [pca_one, pca_two, pca_opt]
    print(pca_opt)
    return new_ratio, var_one, var_two, cor, corrs, pca, cols

# Getting abnomal SVI


def abnormal(column, periods):
    col = [None]*periods
    for i in range(periods, len(column)):
        x = column.iloc[i]
        y = np.median(column.iloc[i-5:i])
        z = math.log(x/y)
        col.append(z)
    return col

# Detreding trading volume


def ratio_tv(columns):
    col = []
    for i in range(len(columns)):
        z = columns.iloc[i, 0]/columns.iloc[i, -1]
        col.append(z)
    return col

# Detrending the attention data


def detrending_one(data):
    dat = pd.DataFrame(data.loc[:, :])
    dat['MTR'] = np.nan
    # dat['ASVI'] = np.nan
    array1 = ratio_tv(data[['TV', 'SHOUT']])
    # array2 = abnormal(data.loc[:,'SVI'],5)
    for i in range(len(dat)):
        dat.iloc[i, -1] = array1[i]
        # dat.iloc[i,-1] = array2[i]
    # dat = dat.drop('SVI',axis = 1)
    dat = dat.drop('TV', axis=1)
    dat = dat.drop('SHOUT', axis=1)
    return dat

# Getting the ratio indexes for the datasat split into sentiment and attention


def get_pca_split_new(rawdata, list1, list2):
    # Attention with option 1
    list_mix = []
    for i in range(len(list1)):
        list_mix.append(list1[i])
    list_mix.append('SHOUT')
    df_at = rawdata[list_mix]
    df_at_de = detrending_one(df_at).iloc[1:, :]
    df_at_de = df_at_de[df_at_de.index >= mindate]
    df_at_de = df_at_de.diff()
    diff_datas_attn.append(df_at_de)
    data_att = standardize(df_at_de.iloc[1:, :])
    sta_datas_att.append(data_att)
    attention, v1_at, v2_at, co_at, corrs_at, pca_a, cols1 = get_ratio(
        data_att, data_att.columns)

# Sentiment
    df_se = rawdata[list2].iloc[1:, :]
    df_se = df_se[df_se.index >= mindate]
    df_se = df_se.diff()
    
    diff_datas_sent.append(df_se)
    data_sent = standardize(df_se.iloc[1:, :])
    sta_datas_sent.append(data_sent)
    sentiment, v1_se, v2_se, co_sent, corrs_se, pca_s, cols2 = get_ratio(
        data_sent, list2)

    vars = [v1_at, v2_at, v1_se, v2_se]
    cor = [f"att: {co_at}", f"sent: {co_sent}"]
    corrs = [corrs_at, corrs_se]
    pcas = [pca_a, pca_s]
    cols = [cols1, cols2]
    return attention, sentiment, vars, cor, corrs, pcas, cols

# Get the dataframe with returns of a given stock and its comparative ratios


def get_returns(data):
    df = pd.DataFrame(data=data.iloc[:, 0])
    df['Returns'] = None
    for i in range(1, len(df)):
        df.iloc[i, -1] = (df.iloc[i, 0]-df.iloc[i-1, 0])/df.iloc[i-1, 0]
    return df

# Combining the data


def combined_data(r, freq, list1, list2):
    frequency = freq
    # rawdata = read_raw(r,frequency)
    '''if freq != "Daily":
        new_list = []
        for i in range(len(list2)):
            if list2[i] != "VMP":
                new_list.append(list2[i])
        list2 = new_list
    '''
    rawdata = read_data(r, frequency)
    ratio_att, ratio_sent, vars, cors, corrs, pca, cols = get_pca_split_new(
        rawdata, list1, list2)
    df = pd.DataFrame(rawdata.iloc[-len(ratio_att):, 0])
    df["Delta Price"] = df['Price'].diff()
    df['ATTN'] = ratio_att
    df['SENT'] = ratio_sent
    df['Log Return'] = np.nan
    df['Delta ATTN'] = df['ATTN'].diff()
    df['Delta SENT'] = df['SENT'].diff()
    df['Return'] = np.nan
    df['%Delta ATTN'] = np.nan
    df['%Delta SENT'] = np.nan
    for i in range(1, len(df)):
        old = df.iloc[i-1, 0]
        new = df.iloc[i, 0]
        df.iloc[i, 4] = math.log(new/old)
        df.iloc[i, -3] = (new-old)/old
        df.iloc[i, -2] = df.iloc[i, 5]/df.iloc[i-1, 2]
        df.iloc[i, -1] = df.iloc[i, 6]/df.iloc[i-1, 3]
    return df, cors, corrs, pca, cols


# Minimum date
mindate = dt.datetime(2015, 2, 1)
mindate = mindate.strftime('%Y-%m-%d')

# Listing all variables
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

# Sentiment
sentiment_list = [tpsc, tnsc, npsc, nnsc, pcr, ivm]
attention_list = [tvl, tpc, npc, svi]

# Frequencies
frequencies = ["Daily", "Weekly", "Monthly"]
x = combined_data(3, "Daily", attention_list, sentiment_list)

# Collecting our data
finaldata = []
finaltickers = []
correlations = []
attn_d = pd.DataFrame(columns=[tpc, npc, svi, "MTR"])
attn_w = pd.DataFrame(columns=attention_list)
attn_m = pd.DataFrame(columns=attention_list)
sent_d = pd.DataFrame(columns=sentiment_list)
sent_w = pd.DataFrame(columns=sentiment_list)
sent_m = pd.DataFrame(columns=sentiment_list)

for i in range(6):
    dfs = []
    datatickers = []
    pca = []
    col = []
    for j in range(len(frequencies)):
        ticker = tickers[i]+" "+frequencies[j]
        print(ticker)
        df, cor, corr, pcas, cols = combined_data(
            i, frequencies[j], attention_list, sentiment_list)
        attn = pcas[0][2]
        sent = pcas[1][2]
        pca.append(attn)
        pca.append(sent)
        col.append(cols[0])
        col.append(cols[1])
        dfs.append(df)
        datatickers.append(ticker)

# Gathering the correlations
        c1 = ticker + " "+cor[0]
        c2 = ticker + " "+cor[1]
        correlations.append(c1)
        correlations.append(c2)
# Appending our data
    attn_d.loc[f"{tickers[i]} Cols"] = col[0]
    attn_d.loc[f"{tickers[i]} PCA"] = np.around(pca[0], 2)
    attn_w.loc[f"{tickers[i]} Cols"] = col[2]
    attn_w.loc[f"{tickers[i]} PCA"] = np.around(pca[2], 2)
    attn_m.loc[f"{tickers[i]} Cols"] = col[4]
    attn_m.loc[f"{tickers[i]} PCA"] = np.around(pca[4], 2)
    sent_d.loc[f"{tickers[i]} Cols"] = col[1]
    sent_d.loc[f"{tickers[i]} PCA"] = np.around(pca[1], 2)
    sent_w.loc[f"{tickers[i]} Cols"] = col[3]
    sent_w.loc[f"{tickers[i]} PCA"] = np.around(pca[3], 2)
    sent_m.loc[f"{tickers[i]} Cols"] = col[5]
    sent_m.loc[f"{tickers[i]} PCA"] = np.around(pca[5], 2)

    finaldata.append(dfs)
    finaltickers.append(datatickers)

with open(directory+'/LatexTables/PCA/pca_a_d.tex', 'w') as tf:
    tf.write(attn_d.to_latex())

with open(directory+'/LatexTables/PCA/pca_a_w.tex', 'w') as tf:
    tf.write(attn_w.to_latex())

with open(directory+'/LatexTables/PCA/pca_a_m.tex', 'w') as tf:
    tf.write(attn_m.to_latex())

with open(directory+'/LatexTables/PCA/pca_s_d.tex', 'w') as tf:
    tf.write(sent_d.to_latex())

with open(directory+'/LatexTables/PCA/pca_s_w.tex', 'w') as tf:
    tf.write(sent_w.to_latex())

with open(directory+'/LatexTables/PCA/pca_s_m.tex', 'w') as tf:
    tf.write(sent_m.to_latex())

for i in range(len(finaldata)):
    filepath = directory+"/SavedData/RatioData/"+tickers[i]+".xlsx"
    save_xls(finaldata[i], finaltickers[i], filepath)

for i in range(len(correlations)):
    print(correlations[i])
