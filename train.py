import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from stockstats import StockDataFrame
import pdb

class stock():
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.load_and_merge()

    def load_price(self):
        path = './Data/Stocks/'+self.symbol+'.csv'
        self.price = pd.read_csv(path, 
        usecols = ['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], 
        index_col = 'Date')
        self.price.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Calculate Technical Indicator
        stock = StockDataFrame.retype(self.price)
        indicator_list = ['rsi_6', 'rsi_12', 'boll', 'high_7_sma', 'close_7_sma', 'close_14_sma', 'close_3_sma']
        for indicator in indicator_list:
            indicator_value = stock[indicator]
            self.price[indicator] = indicator_value
        # Delete Extra Columns
        del self.price['close_-1_s']
        del self.price['close_-1_d']
        del self.price['rs_6']
        del self.price['rs_12']
        
    def load_index(self):
        path = './Data/Stocks/^IXIC.csv'
        self.indexies = pd.read_csv(path, usecols = ['Date', 'Adj Close'], 
        index_col = 'Date')
        self.indexies.columns = ['Nasqad']

    def load_and_merge(self, kept = 450):
        self.load_price()
        self.load_index()

        self.data = self.price.merge(self.indexies, how='outer', left_index=True, right_index=True)
        self.get_up_or_down_class()
        self.data = self.data[-kept-1:-1]
        self.data = self.data.fillna(self.data.mean())

    def get_up_or_down_class(self):
        difference_raw = self.data['close'].diff().to_frame()
        difference = (difference_raw>0).astype(int)
        difference.columns = ['difference']
        self.data = self.data.merge(difference, how='outer', left_index=True, right_index=True)

    # convert an array of values into a dataset matrix
    def __create_dataset(self, dataset, look_back=1, target_column = 3):
        # ['open', 'high', 'low', 'close', 'volume']
        # difference is added in the end as [17]
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), :-1]
            dataX.append(a)
            dataY.append(dataset[i + look_back, target_column]) # Target is the target_column for the next day
        return np.array(dataX), np.array(dataY)

    def train(self, train_test_split = 0.95, verbose=0, plot = False, lookback = 7, epochs = 100):
        data = self.data
        length = len(data)
        number_of_features = data.shape[1]-1
        target = data['close']
        data = (data.values.astype('float32'))

        print(data.shape)
        # Scale the Percentage Change
        scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        target_scaler.fit(target.reshape(-1,1))

        train_size = int(len(data) * train_test_split)
        test_size = len(data) - train_size
        train, test = data[0:train_size,:], data[train_size:len(data),:]

        # Create Dataset
        look_back = lookback
        all_X, all_Y = self.__create_dataset(data, look_back)
        trainX, trainY = self.__create_dataset(train, look_back)
        testX, testY = self.__create_dataset(test, look_back)	

        # print(target_scaler.inverse_transform(trainY.reshape(-1,1)))
        
        # trainX = np.reshape(trainX, (trainX.shape[0], look_back, train.shape[1]))
        # testX = np.reshape(testX, (testX.shape[0], look_back, test.shape[1]))

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, number_of_features), return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        #Step 2 Build Model

        # Train model
        epochs = epochs
        batch_size = 1000

        # model.load_weights('saved_models/weights.test_run')
        path = 'saved_models/weights.stock.'+self.symbol
        # if test_flag: path = 'saved_models/weights.stock.test.'+symbol
        checkpointer = ModelCheckpoint(filepath=path, 
                                        verbose=verbose, save_best_only=True)

        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpointer], verbose = verbose)

        # scaler_filename = "saved_scalers/scaler.save." + symbol
        # if test_flag: scaler_filename = "saved_scalers/scaler.save.test." + symbol
        # joblib.dump(scaler, scaler_filename)

        # Load the best trained model
        model.load_weights(path)

        # Testing accuracy
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # invert predictions and targets to unscaled
        trainPredict = target_scaler.inverse_transform(trainPredict.reshape(-1,1))
        trainY = target_scaler.inverse_transform(trainY.reshape(-1,1))
        testPredict = target_scaler.inverse_transform(testPredict.reshape(-1,1))
        testY = target_scaler.inverse_transform(testY.reshape(-1,1))

        # Report training and test accuracy
        import math
        from sklearn.metrics import mean_squared_error
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        print('Train Score: %.9f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY, testPredict))
        print('Test Score: %.9f RMSE' % (testScore))

        # print(testPredict[-10:-1], testY[-10:-1])

        # # Save scaler
        # scaler_filename = "saved_scalers/scaler.save." + symbol
        # if test_flag: scaler_filename = "saved_scalers/scaler.save.test." + symbol
        # joblib.dump(scaler, scaler_filename)

        # # Return the difference between prediction and ground truth for further adjustment
        # predict_all = model.predict(all_X)
        # error = np.subtract(scaler.inverse_transform(all_Y), scaler.inverse_transform(predict_all))

        # # Save error
        # error_filename = "saved_errors/errors.save." + symbol
        # if test_flag: error_filename = "saved_errors/errors.save.test." + symbol
        # joblib.dump(error, error_filename)

        if plot:
            # Plot train
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(trainY, 'blue')
            plt.plot(trainPredict, 'red')

            plt.subplot(2,1,2)
            plt.plot(testY, 'blue')
            plt.plot(testPredict, 'red')

            plt.show()

    def train_classifier(self, train_test_split = 0.95, verbose=0, plot = False, lookback = 7, epochs = 100):
    
        data = self.data
        target_column = len(data.columns)-1 # The last column is the class column
        length = len(data)
        number_of_features = data.shape[1]-1
        target = data['close']
        data = (data.values.astype('float32'))

        print(data.shape)
        # Scale the Percentage Change
        scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        target_scaler.fit(target.reshape(-1,1))

        train_size = int(len(data) * train_test_split)
        test_size = len(data) - train_size
        train, test = data[0:train_size,:], data[train_size:len(data),:]

        # Create Dataset
        look_back = lookback
        all_X, all_Y = self.__create_dataset(data, look_back, target_column = target_column)
        all_Y = all_Y.astype(int)
        trainX, trainY = self.__create_dataset(train, look_back, target_column = target_column)
        trainY = trainY.astype(int)
        testX, testY = self.__create_dataset(test, look_back, target_column = target_column)
        testY = testY.astype(int)

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, number_of_features), return_sequences=False))
        model.add(Dropout(0.2))
        # model.add(Dense(5,activation='relu'))
        # # model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #Step 2 Build Model

        # Train model
        epochs = epochs
        batch_size = 1000

        # model.load_weights('saved_models/weights.test_run')
        path = 'saved_models/weights.stock.classifier.'+self.symbol
        # if test_flag: path = 'saved_models/weights.stock.test.'+symbol
        checkpointer = ModelCheckpoint(filepath=path, 
                                        verbose=verbose, save_best_only=True)

        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpointer], verbose = verbose)

        # scaler_filename = "saved_scalers/scaler.save." + symbol
        # if test_flag: scaler_filename = "saved_scalers/scaler.save.test." + symbol
        # joblib.dump(scaler, scaler_filename)

        # Load the best trained model
        model.load_weights(path)

        # Testing accuracy
        trainPredict = model.predict_classes(trainX)
        testPredict = model.predict_classes(testX)
        # Reshape the output to 1D array
        trainPredict_reshape = trainPredict.reshape(-1)
        testPredict_reshape = testPredict.reshape(-1)

        # Print result
        print('The Classifier training accuracy is:\n')
        print(sum(trainY==trainPredict_reshape)/len(trainY))
        print('The Classifier testing accuracy is:\n')
        print(sum(testY==testPredict_reshape)/len(testY))

        print(trainY)

        # # Save scaler
        # scaler_filename = "saved_scalers/scaler.save." + symbol
        # if test_flag: scaler_filename = "saved_scalers/scaler.save.test." + symbol
        # joblib.dump(scaler, scaler_filename)

        # # Return the difference between prediction and ground truth for further adjustment
        # predict_all = model.predict(all_X)
        # error = np.subtract(scaler.inverse_transform(all_Y), scaler.inverse_transform(predict_all))

        # # Save error
        # error_filename = "saved_errors/errors.save." + symbol
        # if test_flag: error_filename = "saved_errors/errors.save.test." + symbol
        # joblib.dump(error, error_filename)

        # if plot:
        #     # Plot train
        #     plt.figure()
        #     plt.subplot(2,1,1)
        #     plt.plot(trainY, 'blue')
        #     plt.plot(trainPredict, 'red')

        #     plt.subplot(2,1,2)
        #     plt.plot(testY, 'blue')
        #     plt.plot(testPredict, 'red')

        #     plt.show()

if __name__ == '__main__':
    #['rsi_6', 'rsi_12', 'boll', 'high_7_sma', 'close_7_sma', 'close_14_sma', 'close_3_sma']
    AAPL = stock('AAPL')
    # print(AAPL.data)
    print(AAPL.data)
    print(len(AAPL.data.columns))
    # AAPL.train(verbose = 1, plot=True, lookback = 5, epochs = 7000)
    AAPL.train(verbose = 1, plot=True, lookback = 5, epochs = 1000)

    # To fix a bug in tensorflow with the current environment set up
    from keras import backend as K
    K.clear_session()
