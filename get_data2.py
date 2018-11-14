import pandas as pd
import numpy as np
import pandas_datareader as pdr
in_stock = ['AAPL', 'GOOGL', 'KO', 'GE', 'IBM']
df = pd.DataFrame()
dict_in = {}
for i in range(5):
    df = pdr.get_data_yahoo(in_stock[i])
    df['prices']['Close'].name = in_stock[i]
    dict_in[in_stock[i]] = df['prices']['Close']
data_in = pd.DataFrame(dict_in)
data_in.dropna()
print(data_in.info())

writer = pd.ExcelWriter('stock_in.xlsx')
data_in.to_excel(writer)
writer.save()
