import numpy as np
import pandas as pd
import scipy.optimize as sco
import scipy.interpolate as sci
import matplotlib.pyplot as plt
import tushare as ts

# national and international stock
na_stock = ['000002', '000063', '002415', '000333', '600036']
in_stock = []

df_na = []
df = pd.DataFrame()
dict_na = {}
for i in range(5):
    df = ts.get_h_data(na_stock[i], start = '2014-01-01', end = '2018-04-01')
    df['close'].name = na_stock[i]
    dict_na[na_stock[i]] = df['close']
data_na = pd.DataFrame(dict_na)
data_na.dropna()
print data_na.info()

data_na.to_excel('stock_na.xlsx')

data_na.index = data_na['date'].tolist()
data_na.pop('date')
(data_na/data_na.ix[0]*100).plot(figsize=(10,8), grid=True)
