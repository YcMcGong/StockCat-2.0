import gensim
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

"""
The last date of the news is Nov 13
"""
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True, limit=500000) 

# Reading data
file_path = './Data/news/AAPL.csv'
df = pd.read_csv(file_path)

# Change Dates to be without hours
for i in range(len(df)):
    df['PUBLICATION_DATE'][i]= df['PUBLICATION_DATE'][i][0:10]

""" Generate a dataset with Word <-> Error"""
symbol = 'SNE'
# Loading error
error_filename = "saved_errors/errors.save.SNE.nlp.NOV13"
error = joblib.load(error_filename)
# Loading error scaler
error_scaler_filename = "saved_scalers/scaler.save.error." + symbol
error_scaler = joblib.load(error_scaler_filename)
# Scaled error data
error_scaled = error_scaler.fit_transform(error)
targe_error = []

# In house method to combine data for the same day
def create_concat_titles():
    temp_date = df['PUBLICATION_DATE'][0]
    data = []
    day_index = 0
    error_index = 0
    sentence = ""
    for i in range(len(df)):
        if (df['PUBLICATION_DATE'][i] == temp_date) or df['PUBLICATION_DATE'][i][0:4]!='2017':
            if len(str(df['SUMMARY'][i]))>24:
                sentence = sentence + '. ' + df['SUMMARY'][i]
        else:
            # print(df['PUBLICATION_DATE'][i])
            temp_date = df['PUBLICATION_DATE'][i]
            # Check if weekend
            if calculate_if_weekday(day_index):
                data.append(sentence)
                sentence = df['SUMMARY'][i]
                targe_error.append(error_scaled[1+error_index]) # Append error to the target
            day_index = day_index + 1
            error_index = error_index + 1
    
    data.append(sentence)
    targe_error.append(error_scaled[1+error_index])

    print(len(data))
    return data

def calculate_if_weekday(i):
    # Assume start with Nov 13
    start_day = 1 # Monday
    weekend_index = [start_day, start_day+1]
    day = i%7
    if day in weekend_index:
        return False
    else:
        return True

number_of_feature = 10

def create_feature(text):
    word_list = word_tokenize(text)
    features = []
    for word in word_list:
        try:
            features.append(model[lemmatizer.lemmatize(word)][:number_of_feature])
        except:
            pass
    if len(features)==0: return np.array([0.0]*number_of_feature)
    return np.sum(features, axis = 0)

def create_feature_set(texts):
    feature_set = []
    for text in texts:
        feature_set.append(create_feature(text))
    length = len(feature_set)
    return np.array(feature_set).reshape(length, number_of_feature)

if __name__ == '__main__':
    texts = create_concat_titles()
    data = create_feature_set(texts)
    # print(len(targe_error))
    targe_error = np.array(targe_error).reshape(-1,1)

    # Train test split
    trainX, testX, trainY, testY = train_test_split(
        data, targe_error, test_size=0.7, random_state=42)

    # Create adjust model
    model = Sequential()
    model.add(Dense(units=8, activation=None, input_shape=(number_of_feature,)))
    model.add(Dense(units=1))
    #Step 2 Build Model
    # Compile neural network
    model.compile(loss='mse', # Mean squared error
                optimizer='adam', # Optimization algorithm
                metrics=['mse']) # Mean squared error

    verbose = 1
    epochs = 8000
    path = 'saved_models/weights.stock.nlp.'+symbol
    checkpointer = ModelCheckpoint(filepath=path, 
                                    verbose=verbose, save_best_only=False)

    model.fit(trainX, trainY, nb_epoch=epochs, validation_split=0.05, callbacks=[checkpointer], verbose = verbose)

    # Testing accuracy
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

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

    # Predict all data
    nlp_correction = model.predict(data)
    # Save nlp_correction
    nlp_correction_filename = "saved_nlp_correction/nlp.correction.save." + symbol
    joblib.dump(nlp_correction, nlp_correction_filename)