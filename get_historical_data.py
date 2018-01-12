from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import pandas as pd

yf.pdr_override()
start_date = "2015-11-17"
end_date = "2017-11-17"
stock_list = ['AAPL','PYPL','SNE','NVDA','AVGO','JD','NFLX','^NYA']
for symbol in stock_list:
    data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    filename = './Data/Stocks/' + symbol + '.csv'
    data.to_csv(filename)