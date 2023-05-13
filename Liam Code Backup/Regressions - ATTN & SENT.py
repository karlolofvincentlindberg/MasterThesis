import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.decomposition import PCA
import statsmodels.api as sm

# Inputs:

ticker = "AAPL"
frequency = "Monthly"

# ATTN Import

filepath_attn = rf"C:\Users\liam1\OneDrive\Desktop\Output - Final\{ticker} - Attention Final - {frequency}.csv"
attn = pd.read_csv(filepath_attn)
attn['Date'] = pd.to_datetime(attn['Date'])
attn.set_index('Date', inplace=True)
attn_proxies = ["ATTN U.V."]
columns_to_drop = set(attn.columns) - set(attn_proxies)
attn.drop(columns = columns_to_drop, inplace=True)
print(attn.head())

# SENT Import

filepath_sent = rf"C:\Users\liam1\OneDrive\Desktop\Output - Final\{ticker} - Sentiment Final - {frequency}.csv"
sent = pd.read_csv(filepath_sent)
sent['Date'] = pd.to_datetime(sent['Date'])
sent.set_index('Date', inplace=True)
sent_proxies = ["SENT U.V."]
columns_to_drop = set(sent.columns) - set(sent_proxies)
sent.drop(columns = columns_to_drop, inplace=True)
print(sent.head())

# Price Import 

filepath_price = rf"C:\Users\liam1\OneDrive\Desktop\Data - Final\{ticker} - {frequency}.csv"
price = pd.read_csv(filepath_price)
price['Date'] = pd.to_datetime(price['Date'])
price.set_index('Date', inplace=True)
print(price.head())
price_proxies = ["Price"]
columns_to_drop = set(price.columns) - set(price_proxies)
price.drop(columns = columns_to_drop, inplace=True)
print(price.head())

# Merge ATTN & SENT DataFrames
merged = pd.merge(sent, attn, left_index=True, right_index=True, how='outer')

# Creating Lags

for col in merged:

    merged[f'{col} L1'] = merged[col].shift(1)
    merged[f'{col} L2'] = merged[col].shift(2)  
    merged[f'{col} L3'] = merged[col].shift(3)

merged = merged.reindex(sorted(merged.columns), axis = 1)
merged = merged.dropna(axis=0)

# Merge w/ Price

reg = pd.merge(merged, price, left_index=True, right_index=True, how='outer')
reg = reg.dropna(axis=0)
print(reg.head())
reg.corr()

# Multivariate Regression - Main

independent = list(reg.columns)
independent.remove('Price')
dependent = 'Price'
print(independent)

X = reg[independent]
y = reg[dependent]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# Multivariate Regression - Other

X = reg[['ATTN U.V.']]
y = reg['Price']
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

X = reg[['SENT U.V.']]
y = reg['Price']
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

X = reg[['SENT U.V.', 'ATTN U.V.']]
y = reg['Price']
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())