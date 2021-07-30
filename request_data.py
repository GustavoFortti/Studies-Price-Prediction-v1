import yfinance as yf
import pandas_datareader.data as web
import numpy as np
import pandas as pd
from datetime import datetime

def import_data():
    try :
        data = web.get_data_yahoo("BTC-USD") #, start=datetime(1981, 1, 1))
        data['day'] = data.index
        data = data.set_index(np.arange(0, len(data)))
        data = data.drop(columns=['Adj Close', 'day'])
        
        data = data[:len(data)]

    except :
        print('Request error')
        data = 0

    return data

def pull_data():
    data = import_data()

    data.to_csv('../data/data.csv', index=False)
