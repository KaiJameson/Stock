import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque
import alpaca_trade_api as tradeapi
from api_key import real_api_key_id, real_api_secret_key
from environment import test_var, reports_directory, graph_directory, money
from time_functions import get_time_string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
import datetime
import math 

def make_dataframe(symbol, timeframe='day', limit=1000):
    api = tradeapi.REST(real_api_key_id, real_api_secret_key)
    barset = api.get_barset(symbols=symbol, timeframe='day', limit=limit)
    items = barset.items()
    data = {}
    for symbol, bar in items:
        open_values = []
        close_values = []
        low_values = []
        high_values = []
        mid_values = []
        for day in bar:
            open_price = day.o
            close_price = day.c
            low_price = day.l
            high_price = day.h
            mid_price = (low_price + high_price) / 2
            open_values.append(open_price)
            close_values.append(close_price)
            low_values.append(low_price)
            high_values.append(high_price)
            mid_values.append(mid_price)
        data['open'] = open_values
        data['low'] = low_values
        data['high'] = high_values
        data['close'] = close_values
        data['mid'] = mid_values
    df = pd.DataFrame(data=data)
    return df


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
                test_size=0.2, feature_columns=['open', 'low', 'high', 'close', 'mid']):
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load data from alpaca
        df = make_dataframe(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df[test_var].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    train = tf.data.Dataset.from_tensor_slices((result["X_train"], result["y_train"]))
    test = tf.data.Dataset.from_tensor_slices((result["X_test"], result["y_test"]))
    train = train.batch(64)
    test = test.batch(64)
    # return the result
    return result, train, test


def create_model(sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    #print('i am going to compile this model and return it')
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


def predict(model, data, n_steps, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][:n_steps]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_val = column_scaler[test_var].inverse_transform(prediction)[0][0]
    return predicted_val


def plot_graph(model, data, ticker='default'):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"][test_var].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"][test_var].inverse_transform(y_pred))
    # last 200 days, feel free to edit that
    real_y_values = y_test[-100:]
    predicted_y_values = y_pred[-100:]
    spencer_money = money * (real_y_values[-1]/real_y_values[0])
    file_name = reports_directory + '/' + ticker + '/' + get_time_string() + '.txt'
    f = open(file_name, 'w')
    f.write('spencer wanted me to have: $' + str(spencer_money) + '\n')
    money_made = decide_trades(money, real_y_values, predicted_y_values)
    f.write('money made from using real vs predicted: $' + str(money_made) + '\n')
    f.close()
    plot_dir = graph_directory + '/' + ticker
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    plot_name = plot_dir + '/' + test_var + '_' + get_time_string() + '.png'
    plt.plot(real_y_values, c='b')
    plt.plot(predicted_y_values, c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.savefig(plot_name)
    plt.close()
    return real_y_values[-1]


def get_accuracy(model, data, lookup_step):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"][test_var].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"][test_var].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-lookup_step], y_pred[lookup_step:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-lookup_step], y_test[lookup_step:]))
    return accuracy_score(y_test, y_pred)


def decide_trades(money, data1, data2):
    stocks_owned = 0
    for i in range(1,len(data1)):
        now_price = data1[i-1]
        if data2[i] > now_price:
            stocks_can_buy = money // now_price
            if stocks_can_buy > 0:
                money -= stocks_can_buy * now_price
                stocks_owned += stocks_can_buy
        elif data2[i] < now_price:
            money += now_price * stocks_owned
            stocks_owned = 0
    if stocks_owned != 0:
        money += stocks_owned * data1[len(data1)-1]
    return money


if __name__ == '__main__':
    ticker = 'TSLA'
    load_data(ticker, n_steps=100)

