import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
import server

yf.pdr_override()

start_date = "2013-01-01"
end_date = "2017-10-10"

stock_list = ['SNE','NVDA','AVGO','JD','NFLX']
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back, :])
	return np.array(dataX), np.array(dataY)

def train_base_stock(symbol, start_date, end_date, train_test_split=0.95, test_flag = False, verbose=1):
	# data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date) //Depecated.

	# Construct API Params
	param = {
	'q': symbol, # Stock symbol (ex: "AAPL")
	'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
	'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
	'p': "3Y" # Period (Ex: "1Y" = 1 year)
	}
	data = get_price_data(param) # Getting the current data
	length = len(data)

	number_of_stock = 1
	data_close = (data['Close'].values.astype('float32'))
	prices = np.reshape(data_close, (length,number_of_stock))

	scaler = MinMaxScaler(feature_range=(0, 1))
	prices = scaler.fit_transform(prices)

	train_size = int(len(prices) * train_test_split)
	test_size = len(prices) - train_size
	train, test = prices[0:train_size,:], prices[train_size:len(prices),:]

	# reshape into X=t and Y=t+1
	look_back = 30
	all_X, all_Y = create_dataset(prices, look_back)
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)	

	trainX = np.reshape(trainX, (trainX.shape[0], look_back, train.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], look_back, test.shape[1]))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(50, input_shape=(look_back, number_of_stock)))
	model.add(Dense(number_of_stock))
	model.compile(loss='mse', optimizer='adam')
	#Step 2 Build Model

	# Train model
	epochs = 100
	batch_size = 1000

	# model.load_weights('saved_models/weights.test_run')
	path = 'saved_models/weights.stock.'+symbol
	if test_flag: path = 'saved_models/weights.stock.test.'+symbol
	checkpointer = ModelCheckpoint(filepath=path, 
									verbose=verbose, save_best_only=False)

	model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size, validation_split=0.05, callbacks=[checkpointer], verbose = verbose)

	scaler_filename = "saved_scalers/scaler.save." + symbol
	if test_flag: scaler_filename = "saved_scalers/scaler.save.test." + symbol
	joblib.dump(scaler, scaler_filename)

	# Testing accuracy
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions and targets to unscaled
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform(trainY)
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform(testY)

	# Report training and test accuracy
	import math
	from sklearn.metrics import mean_squared_error
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY, testPredict))
	print('Test Score: %.2f RMSE' % (testScore))

	# Save scaler
	scaler_filename = "saved_scalers/scaler.save." + symbol
	if test_flag: scaler_filename = "saved_scalers/scaler.save.test." + symbol
	joblib.dump(scaler, scaler_filename)

	# Return the difference between prediction and ground truth for further adjustment
	predict_all = model.predict(all_X)
	error = np.subtract(scaler.inverse_transform(all_Y), scaler.inverse_transform(predict_all))

	# Save error
	error_filename = "saved_errors/errors.save." + symbol
	if test_flag: error_filename = "saved_errors/errors.save.test." + symbol
	joblib.dump(error, error_filename)


def train_adjuster_stock(symbol, test_flag = False, verbose=1):

	# Loading error
	error_filename = "saved_errors/errors.save." + symbol
	error = joblib.load(error_filename)

    # A model to adjust the prediction of the LSTM.
	number_of_sample = len(error)
	number_of_features = 5

	# Set up params for finance API
	param = {
    'q': symbol, # Stock symbol (ex: "AAPL")
    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
    'p': "3Y" # Period (Ex: "1Y" = 1 year)
	}
	data = get_price_data(param) # Getting the current data
	data = data[-number_of_sample-1:-1] # Only keeping the data for the length of the errors / Reading a day before

	data_pre_scaled = data.values #returns a numpy array
	scaler = MinMaxScaler(feature_range=(0, 1))
	data_scaled = scaler.fit_transform(data_pre_scaled)
	data = np.array(data_scaled)

	# Save scaler
	scaler_filename = "saved_scalers/scaler.save.adjust." + symbol
	if test_flag: scaler_filename = "saved_scalers/scaler.save.adjust.test." + symbol
	joblib.dump(scaler, scaler_filename)

	# Fit and Save error scaler
	error_scaler = MinMaxScaler(feature_range=(0, 1))
	error_scaled = error_scaler.fit_transform(error)
	error_scaler_filename = "saved_scalers/scaler.save.error." + symbol
	if test_flag: error_scaler_filename = "saved_scalers/scaler.save.error.test." + symbol
	joblib.dump(error_scaler, error_scaler_filename)

	# Train test split
	trainX, testX, trainY, testY = train_test_split(
		data, error_scaled, test_size=0.9, random_state=42)

	# Create adjust model
	adj_model = Sequential()
	adj_model.add(Dense(units=5, activation='relu', input_shape=(number_of_features,)))
	adj_model.add(Dense(units=10, activation='relu'))
	adj_model.add(Dense(units=1))
	#Step 2 Build Model
	# Compile neural network
	adj_model.compile(loss='mse', # Mean squared error
                optimizer='adam', # Optimization algorithm
                metrics=['mse']) # Mean squared error

	# Train model
	epochs = 1800
	batch_size = 1000
	# Create an adjust model
	path = 'saved_models/weights.stock.adjust.'+symbol
	if test_flag: path = 'saved_models/weights.stock.test.adjust.'+symbol
	checkpointer = ModelCheckpoint(filepath=path, 
									verbose=verbose, save_best_only=False)

	adj_model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size, validation_split=0.05, callbacks=[checkpointer], verbose = verbose)

	# Testing accuracy
	trainPredict = adj_model.predict(trainX)
	testPredict = adj_model.predict(testX)

	# Report training and test accuracy
	import math
	from sklearn.metrics import mean_squared_error

	# invert predictions and targets to unscaled
	trainPredict = error_scaler.inverse_transform(trainPredict)
	trainY = error_scaler.inverse_transform(trainY)
	testPredict = error_scaler.inverse_transform(testPredict)
	testY = error_scaler.inverse_transform(testY)

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY, testPredict))
	print('Test Score: %.2f RMSE' % (testScore))

def test_combined_model(symbol, train_test_split = 0.7):

    # Create models
	number_of_stock = 1
	look_back = 30
	number_of_features = 5

	# Create the LSTM network
	model = Sequential()
	model.add(LSTM(50, input_shape=(look_back, number_of_stock)))
	model.add(Dense(number_of_stock))
	model.compile(loss='mse', optimizer='adam')

	# Create adjust model
	adj_model = Sequential()
	adj_model.add(Dense(units=5, activation='relu', input_shape=(number_of_features,)))
	adj_model.add(Dense(units=10, activation='relu'))
	adj_model.add(Dense(units=1))

	# Loading model and scalers
	scaler_filename = "saved_scalers/scaler.save." + symbol
	scaler = joblib.load(scaler_filename)
	model_filename = 'saved_models/weights.stock.' + symbol
	model.load_weights(model_filename)
	# Loading adj model and scalers
	adj_model_filename = 'saved_models/weights.stock.adjust.' + symbol
	adj_model.load_weights(adj_model_filename)
	adj_scaler_filename = "saved_scalers/scaler.save.adjust." + symbol
	adj_scaler = joblib.load(adj_scaler_filename)
	# Loading error scaler
	error_scaler_filename = "saved_scalers/scaler.save.error." + symbol
	error_scaler = joblib.load(error_scaler_filename)

	# Prepare Data
	# Construct API Params
	param = {
	'q': symbol, # Stock symbol (ex: "AAPL")
	'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
	'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
	'p': "3Y" # Period (Ex: "1Y" = 1 year)
	}
	original_data = get_price_data(param) # Getting the current data

	""" Base LSTM model"""
	length = len(original_data)

	number_of_stock = 1
	data_close = (original_data['Close'].values.astype('float32'))
	prices = np.reshape(data_close, (length,number_of_stock))

	# Transform Data
	prices = scaler.transform(prices)

	train_size = int(len(prices) * train_test_split)
	test_size = len(prices) - train_size
	train, test = prices[0:train_size,:], prices[train_size:len(prices),:]

	# reshape into X=t and Y=t+1
	look_back = 30
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)	

	trainX = np.reshape(trainX, (trainX.shape[0], look_back, train.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], look_back, test.shape[1]))

	# Testing accuracy
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions and targets to unscaled
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform(trainY)
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform(testY)

	""" Adjustment model"""
	number_of_sample = len(trainY) + len(testY)
	data = original_data[-number_of_sample-1:-1]

	data_pre_scaled = data.values #returns a numpy array
	data_scaled = adj_scaler.transform(data_pre_scaled)
	data = np.array(data_scaled)

	# Use adj_model to predict difference
	error = adj_model.predict(data)

	# invert predictions and targets to unscaled
	error_unscaled = error_scaler.inverse_transform(error)

	# Add error to LSTM prediction
	trainPredict_adj = np.add(trainPredict, error_unscaled[0:len(trainY)])
	testPredict_adj = np.add(testPredict, error_unscaled[len(trainY):])

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY, trainPredict_adj))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY, testPredict_adj))
	print('Test Score: %.2f RMSE' % (testScore))

	print(testPredict[-1])
	print(testY[-1])
	print(error_unscaled[-1])
	print(testPredict_adj[-1])

if __name__ == '__main__':
	for stock in stock_list:
		error = train_base_stock(stock, start_date, end_date, train_test_split=0.7, test_flag = False, verbose=0)
		train_adjuster_stock(stock, test_flag = False, verbose=0)
		test_combined_model(stock)

	# To fix a bug in tensorflow with the current environment set up
	from keras import backend as K
	K.clear_session()