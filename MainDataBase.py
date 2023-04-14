import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import scipy.linalg as la

from datetime import date
from pandas import ExcelWriter

#Our directory
directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"

#Reading in our monthly data files
amazon_m = directory+"/Data/Monthly/AMZNMonthly.csv"
apple_m = directory+"/Data/Monthly/AAPLMonthly.csv"
facebook_m = directory+"/Data/Monthly/METAMonthly.csv"
google_m = directory+"/Data/Monthly/GOOGLMonthly.csv"
microsoft_m = directory+"/Data/Monthly/MSFTMonthly.csv"
netflix_m = directory+"/Data/Monthly/NFLXMonthly.csv"

#Reading in our weekly data files
amazon_w = directory+"/Data/Weekly/AMZNWeekly.csv"
apple_w = directory+"/Data/Weekly/AAPLWeekly.csv"
facebook_w = directory+"/Data/Weekly/METAWeekly.csv"
google_w = directory+"/Data/Weekly/GOOGLWeekly.csv"
microsoft_w = directory+"/Data/Weekly/MSFTWeekly.csv"
netflix_w = directory+"/Data/Weekly/NFLXWeekly.csv"

#Reading in our daily data files
amazon_d = directory+"/Data/Daily/AMZNDaily.csv"
apple_d = directory+"/Data/Daily/AAPLDaily.csv"
facebook_d = directory+"/Data/Daily/METADaily.csv"
google_d = directory+"/Data/Daily/GOOGLDaily.csv"
microsoft_d = directory+"/Data/Daily/MSFTDaily.csv"
netflix_d = directory+"/Data/Daily/NFLXDaily.csv"

#company names
stocks = ["amazon stock","apple stock","facebook stock","google stock","microsoft stock","netflix stock"]
tickers = ["AMZN","AAPL","META","GOOGL","MSFT", "NFLX"]
new_columns = ['Date','Price','TV','TC','TPC','TNC','NC','NPC','NNC','SI','SIR','VMP','PCR','SVT - T','SVI - S','SVI','SHOUT']

#Monthly data
companies_m = [amazon_m, apple_m, facebook_m, google_m, microsoft_m, netflix_m]
raw_m = []
per_m = []
sta_m = []

#Weekly data
companies_w = [amazon_w, apple_w, facebook_w, google_w, microsoft_w, netflix_w]
raw_w = []
per_w = []
sta_w = []

#Dailyly data
companies_d = [amazon_d, apple_d, facebook_d, google_d, microsoft_d, netflix_d]
raw_d = []
per_d = []
sta_d = []

#Date limits
maxdate = dt.datetime(2021, 12, 31)
maxdate = maxdate.strftime('%Y-%m-%d')
mindate = dt.datetime(2015, 1, 1)
mindate = mindate.strftime('%Y-%m-%d')

#Turning data into dataframe
def read_data(r,dataframe, mindt,maxdt):
    name = dataframe[r]
    stock = stocks[r]
    ticker = tickers[r]
    data = pd.read_csv(name)
    data['Actual Date'] = pd.to_datetime(data['Actual Date'])
    data['Actual Date'] = data['Actual Date'].dt.strftime('%Y-%m-%d')
    list = ['Ghetto Date']#,['IV Midpoint'],['SI'],['SI Ratio']
    for i in list:
        data = data.drop(i, axis=1)
    data = data.sort_values(by=['Actual Date'])

#Midvolatility
    midvol = pd.read_csv(directory+"/Data/MidVol.csv")
    midvol.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    midvol['comp'] = np.nan
    dates = pd.DataFrame(data=data['Actual Date'])

    for i in range(len(midvol)):
        dat = dates[dates['Actual Date'] >= midvol.iloc[i, 0]]
        midvol.iloc[i, 7] = dat.min()

    for i in range(1, len(midvol)):
        if midvol.iloc[i, 7] == midvol.iloc[i - 1, 7]:
            midvol.iloc[i - 1, 7] = 0

    for j in  range(len(data)):
        da = midvol[midvol['comp'] == data.iloc[j,0]]
        if len(da) != 0:
            data.iloc[j, -1] = da.iloc[0,r+1]

#Inserting the put call - ratio
    pcratio = pd.read_csv(directory+"/Data/OptionsData.csv")
    pcratio['Date'] = pd.to_datetime(pcratio['Date'])
    pcratio['Date'] = pcratio['Date'].dt.strftime('%Y-%m-%d')
    data['P/C Ratio'] = ""
    data['fakedate'] = np.nan
    for i in range(len(data)):
        datta = pcratio[pcratio['Date'] <=data.iloc[i,0]]
        datta = datta[datta['Date'] >data.iloc[i-1,0]]
        data.iloc[i,-1] = datta['Date'].max()
    for i in range(len(data)):
        dataa = pcratio[pcratio['Date'] ==data.iloc[i,-1]]
        if len(dataa) != 0:
            x = dataa.iloc[:,r+1].mean()
            data.iloc[i,-2] = x   
    data = data.drop('fakedate',axis = 1)

#Creating new columns for the SVI Data
    data[ticker] = np.nan
    data[stock] = np.nan
    data['SVI'] = np.nan

#Including the SVI data
    svi = pd.read_csv(directory+"/SVIData/SVImanual.csv")
    svi['Date'] = pd.to_datetime(svi['Date'])
    svi['Date'] = svi['Date'].dt.strftime('%Y-%m-%d')

    for j in  range(len(data)):
        if j == 0:
            dti = svi[svi['Date'] <= data.iloc[j, 0]]
        else:
            dti = svi[svi['Date'] > data.iloc[j-1,0]]
            dti = dti[dti['Date'] <= data.iloc[j,0]]
        data.iloc[j,-3] = dti.loc[:,ticker].sum()
        data.iloc[j, -2] = dti.loc[:, stock].sum()
        data.iloc[j, -1] = data.iloc[j, -3] + data.iloc[j, -2]

#Including the shares outstanding
    shout = pd.read_excel(directory+"/Data/shout.xlsx")
    shout['Date'] = pd.to_datetime(shout['Date'])
    shout['Date'] = shout['Date'].dt.strftime('%Y-%m-%d')
    data['shout'] = np.nan   
    shout['comp'] = np.nan
    
    for i in range(len(shout)):
        dat = dates[dates['Actual Date'] >= shout.iloc[i, 0]]
        shout.iloc[i, -1] = dat.min()

    for i in range(1, len(shout)):
        if shout.iloc[i, -1] == shout.iloc[i - 1, -1]:
            shout.iloc[i - 1, -1] = 0

    for j in  range(len(data)):
        da = shout[shout['comp'] == data.iloc[j,0]]
        da = da[da['Ticker Symbol']==ticker]
        if len(da) != 0:
            data.iloc[j, -1] = da.iloc[0,-2]

    data = data[data['Actual Date'] >= mindt]
    data = data[data['Actual Date'] <= maxdt]
    data = data.set_index('Actual Date')
    for i in range(len(data.columns)):
        data = data.rename(columns={data.columns[i]:new_columns[i]})
    return data

#Monthly data
for i in range(len(companies_m)):
    data = read_data(i, companies_m, mindate, maxdate)
    raw_m.append(data)
    
#Weekly data
for i in range(len(companies_w)):
    data = read_data(i, companies_w, mindate, maxdate)
    raw_w.append(data)

#Daily data
for i in range(len(companies_d)):
    data = read_data(i, companies_d, mindate, maxdate)
    raw_d.append(data)

#Function for exporting the dataframe
def save_xls(list_dfs, labels, xls_path):
    with ExcelWriter(xls_path)as writer:
        for i in range(len(list_dfs)):
            list_dfs[i].to_excel(writer, sheet_name=labels[i])

save_xls(raw_m,tickers,directory+"/SavedData/allFilesRawMonthly.xlsx")
save_xls(raw_w,tickers,directory+"/SavedData/allFilesRawWeekly.xlsx")
save_xls(raw_d,tickers,directory+"/SavedData/allFilesRawDaily.xlsx")

'''
##Principle Component Analysis
#Function for to get all the needed information
def pca(i, data_list):
    datan = data_list[i]
    covar = datan.cov(numeric_only=False)
    corr = datan.corr(numeric_only=False)
    eigval, eigvec = la.eig(covar)
    return covar, corr, eigval, eigvec

#Creating the function to add the information to lists 
def getting_pca(list, data, tickers, freq):
    list1 = []
    list2 = []
    cov_list = []
    eigen_list = []
    for i in range(len(list)):
        c1, c2 , e1, e2 = pca(i, data)
        company = tickers[i]
        list1.append(c1)
        list1.append(c2)
        list2.append(e1)
        list2.append(e2)
        cov_list.append(company+" Cov Matrix - "+ freq)
        cov_list.append(company+" Correlation - " + freq)
        eigen_list.append(company+" Eigenvalues - "+ freq)
        eigen_list.append(company+" Eigenvectors - " + freq)
    return list1,list2, cov_list, eigen_list

#Monthly Information
x, y, z, w= getting_pca(companies_m, sta_m, tickers, "Monthly")
cov_corr_m, eigenvalues_vectors_m, cov_tickers_m, eigen_tickers_m = x , y, z, w

#Weekly information
x, y, z, w= getting_pca(companies_w, sta_w, tickers, "Weekly")
cov_corr_w, eigenvalues_vectors_w, cov_tickers_w,eigen_tickers_w = x , y, z, w

#Daily information
x, y, z, w = getting_pca(companies_d, sta_d, tickers, "Daily")
cov_corr_d, eigenvalues_vectors_d, cov_tickers_d, eigen_tickers_d  = x , y, z, w

#Saving our data
save_xls(cov_corr_m, cov_tickers_m, directory+"/Values/CovariancesCorrelationsMonthly.xlsx")
save_xls(cov_corr_w, cov_tickers_w, directory+"/Values/CovariancesCorrelationsWeekly.xlsx")
save_xls(cov_corr_d, cov_tickers_d, directory+"/Values/CovariancesCorrelationsDaily.xlsx")


#Fixing our monthly eigenvalues 
eigenvalues_vectors_df_m = []
for i in range(int(len(eigenvalues_vectors_m)/2)):
    cols = sta_m[i].columns
    list = ["Eigenvalues"]
    for m in range(len(cols)):
        list.append(cols[m])
    list.append("Variance")
    data = pd.DataFrame(index=list,columns = cols)
    data.iloc[0,:]=eigenvalues_vectors_m[2*i].real
    data.iloc[1:-1,:] = eigenvalues_vectors_m[2*i+1]
    for i in range(len(data)):
        for j in range(len(data.iloc[0,:-1])):
            data.iloc[i,j] = round(data.iloc[i,j] ,4)
    tot = data.iloc[0,:].sum()
    var = data.iloc[0,:]/tot
    data.iloc[-1,:] = var
    data = data.sort_values(by=['Eigenvalues'],axis=1,ascending = False)         
    eigenvalues_vectors_df_m.append(data)


#Fixing our weekly eigenvalues
eigenvalues_vectors_df_w = []
for i in range(int(len(eigenvalues_vectors_w)/2)):
    cols = sta_w[i].columns
    list = ["Eigenvalues"]
    for m in range(len(cols)):
        list.append(cols[m])
    list.append("Variance")
    data = pd.DataFrame(index=list,columns = cols)
    data.iloc[0,:]=eigenvalues_vectors_w[2*i].real
    data.iloc[1:-1,:] = eigenvalues_vectors_w[2*i+1]
    for i in range(len(data)):
        for j in range(len(data.iloc[0,:-1])):
            data.iloc[i,j] = round(data.iloc[i,j] ,4)
    tot = data.iloc[0,:].sum()
    var = data.iloc[0,:]/tot
    data.iloc[-1,:] = var
    data = data.sort_values(by=['Eigenvalues'],axis=1,ascending = False)         
    eigenvalues_vectors_df_w.append(data)


#Fixing our daiily eigenvalues
eigenvalues_vectors_df_d = []
for i in range(int(len(eigenvalues_vectors_d)/2)):
    cols = sta_d[i].columns
    list = ["Eigenvalues"]
    for m in range(len(cols)):
        list.append(cols[m])
    list.append("Variance")
    data = pd.DataFrame(index=list,columns = cols)
    data.iloc[0,:]=eigenvalues_vectors_d[2*i].real
    data.iloc[1:-1,:] = eigenvalues_vectors_d[2*i+1]
    for i in range(len(data)):
        for j in range(len(data.iloc[0,:-1])):
            data.iloc[i,j] = round(data.iloc[i,j] ,4)
    tot = data.iloc[0,:].sum()
    var = data.iloc[0,:]/tot
    data.iloc[-1,:] = var
    data = data.sort_values(by=['Eigenvalues'],axis=1,ascending = False)         
    eigenvalues_vectors_df_d.append(data)


#Eigens
eigens = []
eigens_tickers_all = []

for i in range(len(eigenvalues_vectors_df_d)):
    eigens.append(eigenvalues_vectors_df_d[i])
    eigens.append(eigenvalues_vectors_df_w[i])
    eigens.append(eigenvalues_vectors_df_m[i])
    eigens_tickers_all.append(eigen_tickers_d[2*i])
    eigens_tickers_all.append(eigen_tickers_w[2*i])
    eigens_tickers_all.append(eigen_tickers_m[2*i])


save_xls(eigens, eigens_tickers_all, directory+"/Values/Eigens.xlsx")

#Combining lists
cov_corr_all = []
for i in range(int(len(cov_corr_w)/2)):
    cov_corr_all.append(cov_corr_d[2*i])
    cov_corr_all.append(cov_corr_w[2*i])
    cov_corr_all.append(cov_corr_m[2*i])
    cov_corr_all.append(cov_corr_d[2*i+1])
    cov_corr_all.append(cov_corr_w[2*i+1])
    cov_corr_all.append(cov_corr_m[2*i+1])
    cov_corr_all.append(eigenvalues_vectors_df_d[i])
    cov_corr_all.append(eigenvalues_vectors_df_w[i])
    cov_corr_all.append(eigenvalues_vectors_df_m[i])
cov_tickers_all = []
for i in range(int(len(cov_tickers_w)/2)):
    cov_tickers_all.append(cov_tickers_d[2*i])
    cov_tickers_all.append(cov_tickers_w[2*i])
    cov_tickers_all.append(cov_tickers_m[2*i])
    cov_tickers_all.append(cov_tickers_d[2*i+1])
    cov_tickers_all.append(cov_tickers_w[2*i+1])
    cov_tickers_all.append(cov_tickers_m[2*i+1])
    cov_tickers_all.append(eigen_tickers_d[i])
    cov_tickers_all.append(eigen_tickers_w[i])
    cov_tickers_all.append(eigen_tickers_m[i])

save_xls(cov_corr_all, cov_tickers_all, directory+"/Values/CovariancesCorrelations.xlsx")
'''