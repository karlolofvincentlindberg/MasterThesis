import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.decomposition import PCA

# Inputs:

ticker = "AAPL"
frequency = "Monthly"
filepath = rf"C:\Users\liam1\OneDrive\Desktop\Data - Final\{ticker} - {frequency}.csv"

# Data Manipulation

df = pd.read_csv(filepath)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
original_data = df

# Sentiment DataFrame Setup

sent_proxies = ["TPC", "TNC", "NPC", "NNC", "VMP", "PCR"]
columns_to_drop = set(df.columns) - set(sent_proxies)
df.drop(columns = columns_to_drop, inplace=True)

# Remove Incomplete Rows Due To Frequency

print(df.head())

if frequency == "Daily":
    df = df.iloc[19:,:]

if frequency == "Weekly": 
    df = df.iloc[5:,:] 

if frequency == "Monthly":
    df = df.iloc[1:,:] 

# Standardize Data & Generate Lags

df = (df - df.mean()) / (df.std())

for col in df:

    df[f'{col} L1'] = df[col].shift(1)

df = df.reindex(sorted(df.columns), axis = 1)
df = df.iloc[1:, :]

# PCA Loadings - FSS

pca = PCA(n_components=1).fit(df)
pc1_fss_values = pca.components_

pc1_fss = pd.DataFrame(pc1_fss_values[0], columns = ["PC1 FSS"], index = df.columns)
pc1_fss.index.name = 'Variable'
print(pc1_fss)

# FSS Calculation

df['FSS'] = np.matmul(df, pc1_fss_values.T)
fss = df.iloc[:,-1]

# Variable Correlation w/ FSS

corr_values = []
for i in range(len(df.columns)-1):
    corr_values.append(df.iloc[:,i].corr(df.iloc[:,-1]))
corr = pd.DataFrame(corr_values, columns = ["Corr w/ FSS"], index = df.columns[:-1])
corr.index.name = 'Variable'
print(corr)

# Sentiment Index

proxies = []

for i in range(0, len(corr["Corr w/ FSS"]), 2):
    if abs(corr["Corr w/ FSS"][i]) >= abs(corr["Corr w/ FSS"][i+1]):
        proxies.append(corr.loc[corr['Corr w/ FSS'] == corr["Corr w/ FSS"][i]].index[0])

    if abs(corr["Corr w/ FSS"][i]) < abs(corr["Corr w/ FSS"][i+1]):
        proxies.append(corr.loc[corr['Corr w/ FSS'] == corr["Corr w/ FSS"][i+1]].index[0])
print(proxies)

# Creating Dataframe with SENT proxies

columns_to_drop = set(df.columns) - set(proxies)
df.drop(columns = columns_to_drop, inplace=True)

# PCA Loadings - SENT

pca = PCA(n_components=1).fit(df)
pc1_sent_values = pca.components_
pc1_sent = pd.DataFrame(pc1_sent_values[0], columns = ["PC1 SENT"], index = df.columns)
pc1_sent.index.name = 'Variable'
print(pc1_sent)

# SENT Calculation

df['SENT'] = np.matmul(df, pc1_sent_values.T)
sent = df.iloc[:,-1]

# Scaling to Unit Variance

sent_std = df['SENT'].std()
scaled_pc1_sent_values = pc1_sent_values / sent_std
scaled_pc1_sent = pd.DataFrame(scaled_pc1_sent_values[0], columns = ["PC1 SENT U.V."], index = df.columns[:-1])
scaled_pc1_sent.index.name = 'Variable'
print(scaled_pc1_sent)

# SENT U.V. Calculation

df['SENT U.V.'] = np.matmul(df.iloc[:,:-1], scaled_pc1_sent_values.T)
sent_uv = df.iloc[:,-1]

# Sanity Check

sent_uv.corr(sent)
sent_uv.corr(fss)

# Final Output

df['FSS'] = fss
output = df.iloc[:,-3:]
print(output.head())

'''
Uncomment below to export final CSV. 
'''

# output.to_csv(f'{ticker} - Sentiment Final - {frequency}.csv')