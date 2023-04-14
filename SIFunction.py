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

#Reading in our daily data files
amazon_d = directory+"/Data/Daily/AMZNDaily.csv"
apple_d = directory+"/Data/Daily/AAPLDaily.csv"
facebook_d = directory+"/Data/Daily/METADaily.csv"
google_d = directory+"/Data/Daily/GOOGLDaily.csv"
microsoft_d = directory+"/Data/Daily/MSFTDaily.csv"
netflix_d = directory+"/Data/Daily/NFLXDaily.csv"

companies_d = [amazon_d, apple_d, facebook_d, google_d, microsoft_d, netflix_d]

def trans_data(i):
    data = pd.read_csv(companies_d[i])
    for i in range(10,len(data)):
        if data.iloc[i,-3] == 0:
            first_value = data.iloc[i-1,-3]
            last_value = 0
            values = 0
            for j in range(i+1,len(data)):
                values = values + 1
                if  data.iloc[j,-3] != 0:
                    last_value = data.iloc[j,-3]
                    break
            x = (last_value-first_value)/values
            data.iloc[i,-3] = first_value+x
    for i in range(10,len(data)):
        if data.iloc[i,-2] == 0:
            first_value = data.iloc[i-1,-2]
            last_value = 0
            values = 0
            for j in range(i+1,len(data)):
                values = values + 1
                if  data.iloc[j,-2] != 0:
                    last_value = data.iloc[j,-2]
                    break
            x = (last_value-first_value)/values
            data.iloc[i,-2] = first_value+x
    return data

for i in range(len(companies_d)):
    data = trans_data(i)
    data.to_csv(companies_d[i])