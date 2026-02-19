import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt


def get_market_data(tickers, start_date = dt.datetime(2015,1,1), end_date=dt.datetime.today()):
    df = yf.download(tickers, start=start_date, end=end_date,
                     progress=False, auto_adjust=False, multi_level_index=False)
    return df['Adj Close']

def calculate_returns(prices, mode='log'):
    
    if mode == 'log':
        return np.log(prices / prices.shift(1)).dropna()