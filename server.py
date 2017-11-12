import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from datetime import datetime as dt
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
yf.pdr_override()

# number_of_days = 30
stock_list = ['SNE','NVDA','AVGO','JD','NFLX']
start_date = "2017-8-10"

def predict_next_day(symbol, isAdjust = False):
    number_of_days = 30
    number_of_features = 5
    # Construct API Params
    param = {
    'q': symbol, # Stock symbol (ex: "AAPL")
    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
    'p': "1Y" # Period (Ex: "1Y" = 1 year)
    }
    # Getting the current date
    # today = dt.now()
    # end_date = today.strftime('%Y-%m-%d')
    # end_date = '2017-10-25'
    # Getting the stock data from start_date to current date
    # data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    data = get_price_data(param)
    # print(data)
    # print(data==None)
    data = data[-number_of_days:] # Only keeping the data for the past number_of_days days

    # Create a model holder
    model = Sequential()
    model.add(LSTM(50, input_shape=(number_of_days, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # # Loading model and scalers
    # scaler_filename = "saved_scalers/scaler.save." + symbol
    # scaler = joblib.load(scaler_filename)
    # model_filename = 'saved_models/weights.stock.' + symbol
    # model.load_weights(model_filename)

    # Loading model and scalers
    scaler_filename = "saved_scalers/scaler.save." + symbol
    scaler = joblib.load(scaler_filename)
    model_filename = 'saved_models/weights.stock.' + symbol
    model.load_weights(model_filename)

    # Preprocess the data
    length = len(data)
    prices = (data['Close'].values.astype('float32'))
    prices = np.reshape(prices, (length,1))
    prices = scaler.fit_transform(prices)
    prices = np.array([prices])

    # Predict next day
    predict = model.predict(prices)
    next_day_price = scaler.inverse_transform(predict)

    """ Using adjustment model"""
    if isAdjust:

        # Create adjust model
        adj_model = Sequential()
        adj_model.add(Dense(units=5, activation='relu', input_shape=(number_of_features,)))
        adj_model.add(Dense(units=10, activation='relu'))
        adj_model.add(Dense(units=1))

        # Loading adj model and scalers
        adj_model_filename = 'saved_models/weights.stock.adjust.' + symbol
        adj_model.load_weights(adj_model_filename)
        adj_scaler_filename = "saved_scalers/scaler.save.adjust." + symbol
        adj_scaler = joblib.load(adj_scaler_filename)
        # Loading error scaler
        error_scaler_filename = "saved_scalers/scaler.save.error." + symbol
        error_scaler = joblib.load(error_scaler_filename)

        # Using adjustment model
        data_pre_scaled = data.values #returns a numpy array
        data_scaled = adj_scaler.transform(data_pre_scaled)
        data_adj = np.array(data_scaled)

        # Use adj_model to predict difference
        error = adj_model.predict(data_adj[-1].reshape(-1,5))

        # invert predictions and targets to unscaled
        error_unscaled = error_scaler.inverse_transform(error)

        # Add error to LSTM prediction
        next_day_adjusted_price = error_unscaled + next_day_price
        next_day_price = next_day_adjusted_price
    return next_day_price[0][0]

def predict_next_few_days(symbol, days):
    
    number_of_days = 30
    # # Getting the current date
    # # today = dt.now()
    # # end_date = today.strftime('%Y-%m-%d')
    # end_date = '2017-10-31'
    # # Getting the stock data from start_date to current date
    # data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

    # Construct API Params
    param = {
    'q': symbol, # Stock symbol (ex: "AAPL")
    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
    'p': "1Y" # Period (Ex: "1Y" = 1 year)
    }
    data = get_price_data(param) # Getting the current data
    data = data[-number_of_days:] # Only keeping the data for the past number_of_days days

    # Create a model holder
    model = Sequential()
    model.add(LSTM(50, input_shape=(number_of_days, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # Loading model and scalers
    scaler_filename = "saved_scalers/scaler.save." + symbol
    scaler = joblib.load(scaler_filename)
    model_filename = 'saved_models/weights.stock.' + symbol
    model.load_weights(model_filename)
    # error_scaler_filename = "saved_scalers/scaler.save.error." + symbol
    # error_scaler = joblib.load(error_scaler_filename)
    # adj_model_filename = 'saved_models/weights.stock.adjust.' + symbol
    # adj_model.load_weights(adj_model_filename)

    # Preprocess the data
    length = len(data)
    prices = (data['Close'].values.astype('float32'))
    prices = np.reshape(prices, (length,1))
    prices = scaler.fit_transform(prices)
    array_prices = np.array([prices])

    # Predict next few days
    for i in range(days):
        predict = model.predict(array_prices)
        # Add the new predicted price to the stack
        array_prices = np.append([array_prices[0][1:]], np.array(predict).reshape(1,-1,1), axis = 1)
    
    # To fix a bug in tensorflow with the current environment set up
    from keras import backend as K
    K.clear_session()

    # Predict the price in USD
    next_day_price = scaler.inverse_transform(predict)

    return next_day_price

if __name__ == '__main__':
    for symbol in stock_list: 
        print(symbol + '\t price will be:\t', predict_next_day(symbol, isAdjust=True))
        print(symbol + '\t 5 day predictions will be:\t', predict_next_few_days(symbol, 3))
