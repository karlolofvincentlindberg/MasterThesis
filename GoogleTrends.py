
import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt
from datetime import date

directory = "C:/Users/User/OneDrive - CBS - Copenhagen Business School/CBS/MSc in EBA -FIN/MasterThesisPython"

amazon = directory+"/SVIData/AMZNtrends.csv"
apple = directory+"/SVIData/AAPLtrends.csv"
facebook = directory+"/SVIData/METAtrends.csv"
google =  directory+"/SVIData/GOOGLtrends.csv"
microsoft =  directory+"/SVIData/MSFTtrends.csv"
netflix =  directory+"/SVIData/NFLXtrends.csv"

filenames = [amazon, apple, facebook, google, microsoft, netflix]

def turn_data(file):
    data = pd.read_csv(file)
    data_headers = data.columns
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    info = pd.DataFrame(columns=['Date','Ticker','Stock','CumTi','CumSto'])
    for i in range(len(data)):
        if data.iloc[i, 0] == data.iloc[i-1, 0]:
            info.loc[str(data.iloc[i, 0])] = np.nan
            info.iloc[-1, 0] = data.iloc[i, 0]
            info.iloc[-1, 1] = data.iloc[i-1, 1]/data.iloc[i, 1]
            info.iloc[-1, 2] = data.iloc[i-1, 2]/data.iloc[i, 2]
            if len(info) < 2:
                info.iloc[-1, 3] = info.iloc[-1, 1]
                info.iloc[-1, 4] = info.iloc[-1, 2]
            else:
                info.iloc[-1, 3] = info.iloc[-1, 1]*info.iloc[-2, 3]
                info.iloc[-1, 4] = info.iloc[-1, 2] * info.iloc[-2, 4]
    for k in range(len(info)):
        for j in range(len(data)):
            if data.iloc[j, 0] == data.iloc[j-1, 0]:
                data = data.drop(index=data.iloc[j].name)
                break
    data['Match tick'] = np.nan
    data['Match stoc'] = np.nan
    for o in range(len(data)):
        ans = info[info['Date'] < data.iloc[o, 0]]
        if len(ans) == 0:
            data.iloc[o, -2] = 1*data.iloc[o, 1]
            data.iloc[o, -1] = 1*data.iloc[o, 2]
        else:
            matchdate = ans['Date'].max()
            dat = info[info['Date'] == matchdate]
            data.iloc[o, -2] = dat['CumTi']*data.iloc[o, 1]
            data.iloc[o, -1] = dat['CumSto']*data.iloc[o, 2]
    da = data.drop(labels = [data_headers[1], data_headers[2]], axis=1)
    da = da.rename(columns = {da.columns[1]:str(data_headers[1]), da.columns[2]:str(data_headers[2])})
    da = da.set_index('Date')
    return da

data = turn_data(amazon)
dataa = pd.DataFrame( index=data.index)
for i in range(len(filenames)):
    data = turn_data(filenames[i])
    dataa = pd.concat([dataa, data], ignore_index=False, axis=1)

dataa.to_csv(directory+"/SVIData/SVImanual.csv")