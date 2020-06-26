from api_key import real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key
import alpaca_trade_api as tradeapi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque
from environment import test_var, reports_directory, graph_directory, back_test_days, to_plot, test_money, excel_directory, money_per_stock
from time_functions import get_time_string, get_end_date, get_trade_day_back, get_date_string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
import datetime
import math 
import talib as ta


def nn_report(ticker, total_time, model, data, test_acc, valid_acc, train_acc, mae, N_STEPS):
    time_string = get_time_string()
    # predict the future price
    future_price = predict(model, data, N_STEPS)
    
    y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var])

    report_dir = reports_directory + "/" + ticker + "/" + time_string + ".txt"
    reports_folder = reports_directory + "/" + ticker
    if not os.path.isdir(reports_folder):
        os.mkdir(reports_folder)

    if to_plot:
        plot_graph(y_real, y_pred, ticker, back_test_days, time_string)

    total_minutes = total_time / 60

    real_y_values = y_real[-back_test_days:]
    predicted_y_values = y_pred[-back_test_days:]

    curr_price = real_y_values[-1]

    spencer_money = test_money * (curr_price/real_y_values[0])
    f = open(report_dir, "a")
    f.write("~~~~~~~" + ticker + "~~~~~~~\n")
    f.write("Spencer wants to have: $" + str(round(spencer_money, 2)) + "\n")
    money_made = model_money(test_money, real_y_values, predicted_y_values)
    f.write("Money made from using real vs predicted: $" + str(round(money_made, 2)) + "\n")
    per_mon = perfect_money(test_money, real_y_values)
    f.write("Money made from being perfect: $" + str(round(per_mon, 2)) + "\n")
    f.write("The test var was " + test_var + "\n")
    f.write("The mean absolute error is: " + str(round(mae, 4)) + "\n")
    f.write("Total time to run was: " + str(round(total_minutes, 2)) + " minutes.\n")
    f.write("The price at run time was: " + str(round(curr_price, 2)) + "\n")
    f.write("The predicted price for tomorrow is: " + str(future_price) + "\n")
    
    percent = future_price / curr_price
    if curr_price < future_price:
        f.write("That would mean a growth of: " + str(round((percent - 1) * 100, 2)) + "%\n")
        f.write("I would buy this stock.\n")
    elif curr_price > future_price:
        f.write("That would mean a loss of: " + str(abs(round((percent - 1) * 100, 2))) + "%\n")
        f.write("I would sell this stock.\n")
    
    f.write("The average away from the real is: " + str(percent_from_real(y_real, y_pred)) + "%\n")
    f.write("Test accuracy score: " + str(round(test_acc * 100, 2)) + "%\n")
    f.write("Validation accuracy score: " + str(round(valid_acc * 100, 2)) + "%\n")
    f.write("Training accuracy score: " + str(round(train_acc * 100, 2)) + "%\n")
    f.close()

    excel_output(curr_price, future_price)

    return percent

def make_excel_file():
    date_string = get_date_string()

    freal = open(excel_directory + "/" + date_string + "real" + ".txt", "r")
    real_vals = freal.read()
    freal.close()

    fpred = open(excel_directory + "/" + date_string + "predict" + ".txt", "r")
    pred_vals = fpred.read()
    fpred.close()

    f = open(excel_directory + "/" + date_string + ".txt", "a+")
    
    f.write(str(real_vals) + "\n")
    f.write(str(pred_vals))

    f.close()
    
    

def excel_output(real_price, predicted_price):
    date_string = get_date_string()
    f = open(excel_directory + "/" + date_string + "real" + ".txt", "a")
    f.write(str(round(real_price, 2)) + "\t")
    f.close()

    f = open(excel_directory + "/" + date_string + "predict" + ".txt", "a")
    f.write(str(round(predicted_price, 2)) + "\t")
    f.close()


def percent_from_real(y_real, y_predict):
    the_diffs = []
    for i in range(len(y_real) - 1):
        per_diff = (abs(y_real[i] - y_predict[i])/y_real[i]) * 100
        the_diffs.append(per_diff)
    pddf = pd.DataFrame(data=the_diffs)
    pddf = pddf.values
    return round(pddf.mean(), 2)

def make_dataframe(symbol, timeframe="day", limit=1000, time=None, end_date=None):
    api = tradeapi.REST(real_api_key_id, real_api_secret_key)
    if end_date is not None:
        if limit > 1000:
            barset = api.get_barset(symbols=symbol, timeframe="day", limit=1000, until=end_date)
            items = barset.items() 
            df = get_values(items)
            new_end_date = get_trade_day_back(end_date, limit-1000)
            other_barset = api.get_barset(symbols=symbol, timeframe="day", limit=limit-1000, end=new_end_date)
            other_df = get_values(other_barset.items()) 
        else:
            barset = api.get_barset(symbols=symbol, timeframe="day", limit=limit, until=end_date)
            items = barset.items() 
            df = get_values(items)
    else:
        if limit > 1000:
            barset = api.get_barset(symbols=symbol, timeframe="day", limit=1000)
            items = barset.items() 
            df = get_values(items)
            new_end_date = get_trade_day_back(get_end_date(), limit-1000)
            other_barset = api.get_barset(symbols=symbol, timeframe="day", limit=limit-1000, end=new_end_date)
            other_df = get_values(other_barset.items()) 
        else:
            barset = api.get_barset(symbols=symbol, timeframe="day", limit=limit)
            items = barset.items() 
            df = get_values(items)
    
    
    if limit > 1000:
        frames = [other_df, df]
        df = pd.concat(frames) 

    # roll = df.close.rolling(window=10).mean()
    
    # df["rolling_avg"] = roll
    
    # upperband, middleband, lowerband = ta.BBANDS(df.close, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
    # df["upper_band"] = upperband
    # df["lower_band"] = lowerband

    # on_bal_vol = ta.OBV(df.close, df.volume)
    # df["OBV"] = on_bal_vol

    # print(df.high)
    # print(df.low)

    parbol_SAR = ta.SAR(df.high, df.low, .02, .018)
    df["SAR"] = parbol_SAR
    print(df)
    return df


def get_values(items):
    data = {}
    for symbol, bar in items:
        open_values = []
        close_values = []
        low_values = []
        high_values = []
        mid_values = []
        volume = []
        for day in bar:
            open_price = day.o
            close_price = day.c
            low_price = day.l
            high_price = day.h
            mid_price = (low_price + high_price) / 2
            vol = day.v
            open_values.append(open_price)
            close_values.append(close_price)
            low_values.append(low_price)
            high_values.append(high_price)
            mid_values.append(mid_price)
            volume.append(vol)
        data["open"] = open_values
        data["low"] = low_values
        data["high"] = high_values
        data["close"] = close_values
        data["mid"] = mid_values
        data["volume"] = volume
    df = pd.DataFrame(data=data)
    return df

# , "rolling_avg"
def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, test_size=0.2, 
feature_columns=["open", "low", "high", "close", "mid", "volume", "SAR"],
                batch_size=64, end_date=None):
    if isinstance(ticker, str):
        # load data from alpaca
        if end_date is not None:
            df = make_dataframe(ticker, limit=2000, end_date=end_date)
        else:
            df = make_dataframe(ticker, limit=2000)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result["df"] = df.copy()
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
    df["future"] = df[test_var].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df["future"].values):
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
    result["last_sequence"] = last_sequence
    # construct the X"s and y"s
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
    result["X_train"], result["X_valid"], result["y_train"], result["y_valid"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    result["X_valid"], result["X_test"], result["y_valid"], result["y_test"] = train_test_split(result["X_valid"], result["y_valid"], test_size=.5, shuffle=shuffle)   
    
    train = Dataset.from_tensor_slices((result["X_train"], result["y_train"]))
    valid = Dataset.from_tensor_slices((result["X_valid"], result["y_valid"]))
    test = Dataset.from_tensor_slices((result["X_test"], result["y_test"]))
    
    train = train.batch(batch_size)
    valid = valid.batch(batch_size)
    test = test.batch(batch_size)

    # train = train.prefetch(buffer_size=AUTOTUNE)
    # test = test.prefetch(buffer_size=AUTOTUNE)

    # return the result
    return result, train, valid, test


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

def getOwnedStocks():
    api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")
    positions = api.list_positions()
    owned = {}
    for position in positions:
        owned[position.symbol] = position.qty
    return owned

def decide_trades(symbol, owned, accuracy, percent):
    api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")
    try:
        qty = owned[symbol]
        if percent < 1:
            sell = api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day"
            )
            print("\nSELLING:", sell)
            print("\n\n")
    except KeyError:
        if accuracy >= .65:
            if percent > 1:
                barset = api.get_barset(symbol, "day", limit=1)
                current_price = 0
                for symbol, bars in barset.items():
                    for bar in bars:
                        current_price = bar.c
                if current_price == 0:
                    print("\n\nSOMETHING WENT WRONG AND COULDNT GET CURRENT PRICE\n\n")
                else:
                    buy_qty = money_per_stock // current_price
                    buy = api.submit_order(
                        symbol=symbol,
                        qty=buy_qty,
                        side="buy",
                        type="market",
                        time_in_force="day"
                    )
                    print("\nBUYING:", buy)
                    print("\n\n")
    except:
        f = open(error_file, "a")
        f.write("problem with configged stock: " + symbol + "\n")
        exit_info = sys.exc_info()
        f.write(str(exit_info[1]) + "\n")
        traceback.print_tb(tb=exit_info[2], file=f)
        f.close()
        print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

def plot_graph(y_real, y_pred, ticker, back_test_days, time_string):
    real_y_values = y_real[-back_test_days:]
    predicted_y_values = y_pred[-back_test_days:]
    
    plot_dir = graph_directory + "/" + ticker
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    plot_name = plot_dir + "/" + test_var + "_" + get_time_string() + ".png"
    plt.plot(real_y_values, c="b")
    plt.plot(predicted_y_values, c="r")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title(ticker)
    plt.legend(["Actual Price", "Predicted Price"])
    plt.savefig(plot_name)
    plt.close()
    
def get_all_accuracies(model, data, lookup_step):
    y_train_real, y_train_pred = return_real_predict(model, data["X_train"], data["y_train"], data["column_scaler"][test_var])
    train_acc = get_accuracy(y_train_real, y_train_pred, lookup_step)
    y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], data["column_scaler"][test_var])
    valid_acc = get_accuracy(y_valid_real, y_valid_pred, lookup_step)
    y_test_real, y_test_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var])
    test_acc = get_accuracy(y_test_real, y_test_pred, lookup_step)

    return train_acc, valid_acc, test_acc 

def get_accuracy(y_real, y_pred, lookup_step):
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_real[:-lookup_step], y_pred[lookup_step:]))
    y_real = list(map(lambda current, future: int(float(future) > float(current)), y_real[:-lookup_step], y_real[lookup_step:]))
    return accuracy_score(y_real, y_pred)

def return_real_predict(model, X_data, y_data, column_scaler):
    y_pred = model.predict(X_data)
    y_real = np.squeeze(column_scaler.inverse_transform(np.expand_dims(y_data, axis=0)))
    y_pred = np.squeeze(column_scaler.inverse_transform(y_pred))
    return y_real, y_pred


def model_money(money, data1, data2):
    stocks_owned = 0
    for i in range(0 , len(data1) - 1):
        now_price = data1[i]
        predict_price = data2[i + 1]
        if predict_price > now_price:
            stocks_can_buy = money // now_price
            if stocks_can_buy > 0:
                money -= stocks_can_buy * now_price
                stocks_owned += stocks_can_buy
        elif predict_price < now_price:
            money += now_price * stocks_owned
            stocks_owned = 0
    if stocks_owned != 0:
        money += stocks_owned * data1[len(data1)-1]
    return money

def perfect_money(money, data):
    stonks_owned = 0;
    for i in range(0, len(data) - 1):
        now_price = data[i]
        tommorow_price = data[i + 1]
        if tommorow_price > now_price:
            stonks_can_buy = money // now_price
            if stonks_can_buy > 0:
                money -= stonks_can_buy * now_price
                stonks_owned += stonks_can_buy
        elif tommorow_price < now_price:
            money += now_price * stonks_owned
            stonks_owned = 0
    if stonks_owned != 0:
        money += stonks_owned * data[len(data) - 1]
    return money

if __name__ == "__main__":
    
    real_price = ([ 110, 100, 110, 100, 110, 100,])
    predict =([ 110, 100, 110, 100, 110, 100])
   
    money = 100
    print(str(model_money(money, real_price, predict)))
    print(str(perfect_money(money, real_price)))

