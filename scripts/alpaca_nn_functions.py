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
from environment import (test_var, reports_directory, current_price_directory, graph_directory, back_test_days, to_plot, 
test_money, excel_directory, stocks_traded, error_file, load_run_excel, using_all_accuracies)
from time_functions import get_time_string, get_end_date, get_date_string, zero_pad_date_string
from functions import deleteFiles
from symbols import trading_real_money
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import time as timey
import os
import sys
import traceback
import random
import datetime
import math 
import talib as ta
import xgboost as xgb


def nn_report(ticker, total_time, model, data, test_acc, valid_acc, train_acc, N_STEPS):
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
    f.write("Total run time was: " + str(round(total_minutes, 2)) + " minutes.\n")
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

    excel_output(ticker, curr_price, future_price)

    return percent, future_price

def make_excel_file():
    date_string = get_date_string()

    fsym = open(excel_directory + "/" + date_string + "symbol" + ".txt", "r")
    sym_vals = fsym.read()
    fsym.close()

    freal = open(excel_directory + "/" + date_string + "real" + ".txt", "r")
    real_vals = freal.read()
    freal.close()

    fpred = open(excel_directory + "/" + date_string + "predict" + ".txt", "r")
    pred_vals = fpred.read()
    fpred.close()

    f = open(excel_directory + "/" + date_string + ".txt", "a+")
    
    f.write(sym_vals + "\n")
    f.write(str(real_vals) + "\n")
    f.write(str(pred_vals))
    f.close()

    os.remove(excel_directory + "/" + date_string + "symbol" + ".txt")
    os.remove(excel_directory + "/" + date_string + "real" + ".txt")
    os.remove(excel_directory + "/" + date_string + "predict" + ".txt")
    
def make_current_price(curr_price):
    date_string = get_date_string()

    f = open(current_price_directory + "/" + date_string + ".txt", "a")
    f.write(str(round(curr_price, 2)) + "\t")
    f.close()

def excel_output(symbol, real_price, predicted_price):
    date_string = get_date_string()

    f = open(excel_directory + "/" + date_string + "symbol" + ".txt", "a")
    f.write(symbol + ":" + "\t")
    f.close()

    f = open(excel_directory + "/" + date_string + "real" + ".txt", "a")
    f.write(str(round(real_price, 2)) + "\t")
    f.close()

    f = open(excel_directory + "/" + date_string + "predict" + ".txt", "a")
    f.write(str(round(predicted_price, 2)) + "\t")
    f.close()

def make_load_run_excel(symbol, train_acc, valid_acc, test_acc, from_real, percent_away):
    date_string = get_date_string()
    f = open(load_run_excel + "/" + date_string + ".txt", "a")
    f.write(symbol + "\t" + str(round(train_acc * 100, 2)) + "\t" + str(round(valid_acc * 100, 2)) + "\t" 
    + str(round(test_acc * 100, 2)) + "\t" + str(round(from_real, 2)) + "\t" + str(round(percent_away, 2)) 
    + "\n")
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
    api = get_api()

    if end_date is not None:
        df = api.polygon.historic_agg_v2(symbol, 1, "day", _from="2000-01-01", to=end_date).df
    else:
        time_now = zero_pad_date_string()
        df = api.polygon.historic_agg_v2(symbol, 1, "day", _from="2000-01-01", to=time_now).df
  
    df["mid"] = (df.low + df.high) / 2
    df = df.tail(limit)

    # df["7_moving_avg"] = df.close.rolling(window=7).mean()
    
    # upperband, middleband, lowerband = ta.BBANDS(df.close, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
    # df["upper_band"] = upperband
    # df["lower_band"] = lowerband

    # df["OBV"] = ta.OBV(df.close, df.volume)

    # df["relative_strength_index"] = ta.RSI(df.close)
    
    # df["linear_regression"] = ta.LINEARREG(df.close, timeperiod=14)

    # df["linear_regression_angle"] = ta.LINEARREG_ANGLE(df.close, timeperiod=14)

    # df["linear_regression_intercept"] = ta.LINEARREG_INTERCEPT(df.close, timeperiod=14)

    # df["linear_regression_slope"] = ta.LINEARREG_SLOPE(df.close, timeperiod=14)

    # df["pearson's_correlation"] = ta.CORREL(df.high, df.low, timeperiod=30)

    # df["money_flow_index"] = ta.MFI(df.high, df.low, df.close, df.volume, timeperiod=14)

    # df["williams_r"] = ta.WILLR(df.high, df.low, df.close, timeperiod=14)

    # df["standard_deviation"] = ta.STDDEV(df.close, timeperiod=5, nbdev=1)

    # minimum, maximum = ta.MINMAX(df.close, timeperiod=30)
    # df["minimum"] = minimum
    # df["maximum"] = maximum

    # df["commodity_channel_index"] = ta.CCI(df.high, df.low, df.close, timeperiod=14)

    # df["parabolic_SAR"] = ta.SAR(df.high, df.low)

    # df["parabolic_SAR_extended"] = ta.SAREXT(df.high, df.low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, 
    # accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    # df["rate_of_change"] = ta.ROC(df.close, timeperiod=10)

    # df["ht_dcperiod"] = ta.HT_DCPERIOD(df.close)

    # df["ht_trendmode"] = ta.HT_TRENDMODE(df.close)

    # df["ht_dcphase"] = ta.HT_DCPHASE(df.close)

    # df["ht_inphase"], df["quadrature"] = ta.HT_PHASOR(df.close)

    # df["ht_sine"], df["ht_leadsine"] = ta.HT_SINE(df.close)

    # df["ht_trendline"] = ta.HT_TRENDLINE(df.close)

    # df["momentum"] = ta.MOM(df.close, timeperiod=10)

    # df["abs_price_osc"] = ta.APO(df.close, fastperiod=12, slowperiod=26, matype=0)

    # df["KAMA"] = ta.KAMA(df.close, timeperiod=30)

    # df["typical_price"] = ta.TYPPRICE(df.high, df.low, df.close)

    # df["ultimate_osc"] = ta.ULTOSC(df.high, df.low, df.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # df["chaikin_line"] = ta.AD(df.high, df.low, df.close, df.volume)

    # df["chaikin_osc"] = ta.ADOSC(df.high, df.low, df.close, df.volume, fastperiod=3, slowperiod=10)

    # df["norm_average_true_range"] = ta.NATR(df.high, df.low, df.close, timeperiod=14)

    # df["median_price"] = ta.MEDPRICE(df.high, df.low)

    # df["variance"] = ta.VAR(df.close, timeperiod=5, nbdev=1)

    # df["aroon_down"], df["aroon_up"] = ta.AROON(df.high, df.low, timeperiod=14)

    # df["aroon_osc"] = ta.AROONOSC(df.high, df.low, timeperiod=14)

    # df["balance_of_power"] = ta.BOP(df.open, df.high, df.low, df.close)

    # df["chande_momen_osc"] = ta.CMO(df.close, timeperiod=14)

    # df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)

    # df["control_MACD"], df["control_MACD_signal"], df["control_MACD_hist"] = ta.MACDEXT(df.close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

    # df["fix_MACD"], df["fix_MACD_signal"], df["fix_MACD_hist"] = ta.MACDFIX(df.close, signalperiod=9)

    # df["minus_directional_ind"] = ta.MINUS_DI(df.high, df.low, df.close, timeperiod=14)

    # df["minus_directional_move"] = ta.MINUS_DM(df.high, df.low, timeperiod=14)

    # df["plus_directional_ind"] = ta.PLUS_DI(df.high, df.low, df.close, timeperiod=14)

    # df["plus_directional_move"] = ta.PLUS_DM(df.high, df.low, timeperiod=14)

    # df["percentage_price_osc"] = ta.PPO(df.close, fastperiod=12, slowperiod=26, matype=0)

    df["stochas_fast_k"], df["stochas_fast_d"] = ta.STOCHF(df.high, df.low, df.close, fastk_period=5, fastd_period=3, fastd_matype=0)

    # df["stochas_relative_strength_k"], df["stochas_relative_strength_d"] = ta.STOCHRSI(df.close, fastk_period=5, fastd_period=3, fastd_matype=0)

    # df["stochas_slowk"], df["stochas_slowd"] = ta.STOCH(df.high, df.low, df.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # df["TRIX"] = ta.TRIX(df.close, timeperiod=30)

    # df["weighted_moving_avg"] = ta.WMA(df.close, timeperiod=30)

    # df["upband"], df["midband"], df["lowband"] = ta.BBANDS(df.close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

    # df["double_exponetial_moving_avg"] = ta.DEMA(df.close, timeperiod=30)

    # df["exponential_moving_avg"] = ta.EMA(df.close, timeperiod=30)

    # df["MESA_mama"], df["MESA_fama"] = ta.MAMA(df.close)

    # df["midpoint"] = ta.MIDPOINT(df.close, timeperiod=14)

    # df["midprice"] = ta.MIDPRICE(df.high, df.low, timeperiod=14)

    # df["triple_exponential_moving_avg"] = ta.TEMA(df.close, timeperiod=30)

    # df["triangular_moving_average"] = ta.TRIMA(df.close, timeperiod=30)

    # df["avg_directional_movement_index"] = ta.ADX(df.high, df.low, df.close, timeperiod=14)

    # df["true_range"] = ta.TRANGE(df.high, df.low, df.close)

    # df["average_price"] = ta.AVGPRICE(df.open, df.high, df.low, df.close)

    # df["weighted_close_price"] = ta.WCLPRICE(df.high, df.low, df.close)

    # df["beta"] = ta.BETA(df.high, df.low, timeperiod=5)

    # df["time_series_forecast"] = ta.TSF(df.close, timeperiod=14)

    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(df.head(1))
    # print(df.tail(5))

    # df = convert_date_values(df)

    # get_feature_importance(df)
    print(df)
    return df

def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, test_size=0.2, 
feature_columns=["open", "low", "high", "close", "mid", "volume", "stochas_fast_k", "stochas_fast_d"],
                batch_size=64, end_date=None):
    if isinstance(ticker, str):
        # load data from alpaca
        if end_date is not None:
            df = make_dataframe(ticker, limit=4000, end_date=end_date)
        else:
            df = make_dataframe(ticker, limit=4000)

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
    result["X_valid"], result["X_test"], result["y_valid"], result["y_test"] = train_test_split(result["X_valid"], result["y_valid"], test_size=.006, shuffle=shuffle)

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
    api = get_api()
    positions = api.list_positions()
    owned = {}
    for position in positions:
        owned[position.symbol] = position.qty
    return owned

def decide_trades(symbol, owned, accuracy, percent, api_id, api_key):
    api = get_api()
    clock = api.get_clock()
    if clock.is_open:
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

                print("\n~~~SELLING " + sell.symbol + "~~~")
                print("Quantity: " + sell.qty)
                print("Status: " + sell.status)
                print("Type: " + sell.type)
                print("Time in force: "  + sell.time_in_force + "\n\n")
            else:
                print("\n~~~Holding " + symbol + "~~~")

        except KeyError:
            if accuracy >= .5:
                if percent > 1:
                    account_equity = api.get_account().equity
                    barset = api.get_barset(symbol, "day", limit=1)
                    current_price = 0
                    for symbol, bars in barset.items():
                        for bar in bars:
                            current_price = bar.c
                    if current_price == 0:
                        print("\n\nSOMETHING WENT WRONG AND COULDNT GET CURRENT PRICE\n\n")
                    else:
                        buy_qty = (float(account_equity) / stocks_traded) // current_price
                        buy = api.submit_order(
                            symbol=symbol,
                            qty=buy_qty,
                            side="buy",
                            type="market",
                            time_in_force="day"
                        )
                    print("\n~~~Buying " + buy.symbol + "~~~")
                    print("Quantity: " + buy.qty)
                    print("Status: " + buy.status)
                    print("Type: " + buy.type)
                    print("Time in force: "  + buy.time_in_force + "\n\n")

        except:
            f = open(error_file, "a")
            f.write("Problem with configged stock: " + symbol + "\n")
            exit_info = sys.exc_info()
            f.write(str(exit_info[1]) + "\n")
            traceback.print_tb(tb=exit_info[2], file=f)
            f.close()
            print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")
    else:
        print("You tried to trade while the market was closed! You're either ")
        print("testing or stupid. Good thing I'm here!")

def buy_all_at_once(symbols, owned, price_list):
    api = get_api()
    clock = api.get_clock()
    if not clock.is_open:
        print("\nThe market is closed right now, go home. You're drunk.")
        return


    buy_list = []
    for symbol in symbols:
        try:
            barset = api.get_barset(symbol, "day", limit=1)
            current_price = 0
            for symbol, bars in barset.items():
                for bar in bars:
                    current_price = bar.c
            if current_price < price_list[symbol]:
                if symbol not in owned:
                    buy_list.append(symbol)
                
            else:
                if symbol in owned:
                    qty = owned[symbol]

                    sell = api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )

                    print("\n~~~SELLING " + sell.symbol + "~~~")
                    print("Quantity: " + sell.qty)
                    print("Status: " + sell.status)
                    print("Type: " + sell.type)
                    print("Time in force: "  + sell.time_in_force + "\n")

            print("The current price for " + symbol + " is: " + str(round(current_price, 2)))
            make_current_price(current_price)

        except:
            f = open(error_file, "a")
            f.write("Problem with configged stock: " + symbol + "\n")
            exit_info = sys.exc_info()
            f.write(str(exit_info[1]) + "\n")
            traceback.print_tb(tb=exit_info[2], file=f)
            f.close()
            print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

            
    print("The Owned list: " + str(owned))
    print("The buy list: " + str(buy_list))

    account_equity = float(api.get_account().equity)
    buy_power = float(api.get_account().cash)

    value_in_stocks = 1 - (buy_power / account_equity)

    print("Value in stocks: " + str(value_in_stocks))
    print("Account equity: " + str(account_equity))

    stock_portion_adjuster = 0

    if value_in_stocks > .6:
        stock_portion_adjuster = len(buy_list)
    elif value_in_stocks > .3:
        if (len(buy_list) / stocks_traded) > .8:
            stock_portion_adjuster = len(buy_list)
        elif (len(buy_list) / stocks_traded) > .6:
            stock_portion_adjuster = len(buy_list) / .95  # want 95%
        elif (len(buy_list) / stocks_traded) > .4:
            stock_portion_adjuster = len(buy_list) / .90 # want 90%
        else:
            stock_portion_adjuster = len(buy_list) / .80 # want 80%
    else:
        if (len(buy_list) / stocks_traded) > .8:
            stock_portion_adjuster = len(buy_list)
        elif (len(buy_list) / stocks_traded) > .6:
            stock_portion_adjuster = len(buy_list) / .90 # want 90%
        elif (len(buy_list) / stocks_traded) > .4:
            stock_portion_adjuster = len(buy_list) / .75 # want 75%
        else:
            stock_portion_adjuster = len(buy_list) / .65 # want 65%
            

    print("\nThe value in stocks is " + str(value_in_stocks))
    print("The Stock portion adjuster is " + str(stock_portion_adjuster))

    for symbol in symbols:
        try:
            if symbol not in owned and symbol not in buy_list:
                print("~~~Not buying " + symbol + "~~~")
                continue

            elif symbol in owned and symbol not in buy_list:
                print("~~~Holding " + symbol + "~~~")
                continue
            
            else:
                current_price = 0
                barset = api.get_barset(symbol, "day", limit=1)
                for symbol, bars in barset.items():
                    for bar in bars:
                        current_price = bar.c
                buy_qty = (buy_power / stock_portion_adjuster) // current_price

                if buy_qty == 0:
                    print("Not enough money to purchase stock " + symbol + ".")
                    continue

                buy = api.submit_order(
                    symbol=symbol,
                    qty=buy_qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                
                print("\n~~~Buying " + buy.symbol + "~~~")
                print("Quantity: " + buy.qty)
                print("Status: " + buy.status)
                print("Type: " + buy.type)
                print("Time in force: "  + buy.time_in_force + "\n")
                
        except:
            f = open(error_file, "a")
            f.write("Problem with configged stock: " + symbol + "\n")
            exit_info = sys.exc_info()
            f.write(str(exit_info[1]) + "\n")
            traceback.print_tb(tb=exit_info[2], file=f)
            f.close()
            print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

def get_api():
    if trading_real_money:
        api = tradeapi.REST(real_api_key_id, real_api_secret_key, base_url="https://api.alpaca.markets")
    else:
        api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")

    return api

def get_feature_importance(df):
    data = df.copy()
    y = data["close"]
    X = data
   
    train_samples = int(X.shape[0] * 0.8)
 
    X_train_FI = X.iloc[:train_samples]
    X_test_FI = X.iloc[train_samples:]

    y_train_FI = y.iloc[:train_samples]
    y_test_FI = y.iloc[train_samples:]

    regressor = xgb.XGBRegressor(gamma=0.0, n_estimators=150, base_score=0.7, colsample_bytree=1, learning_rate=0.05)
    
    xgbModel = regressor.fit(X_train_FI, y_train_FI, eval_set = [(X_train_FI, y_train_FI), 
    (X_test_FI, y_test_FI)], verbose=False)
    
    fig = plt.figure(figsize=(8,8))
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), 
    tick_label=X_test_FI.columns)
    plt.title('Figure 6: Feature importance of the technical indicators.')
    plt.show()

    feature_names = list(X.columns)
    i = 0
    for feature in xgbModel.feature_importances_.tolist():
        print(feature_names[i], end="")
        print(": "+ str(feature))
        i += 1


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
    if using_all_accuracies:
        y_train_real, y_train_pred = return_real_predict(model, data["X_train"], data["y_train"], data["column_scaler"][test_var])
        train_acc = get_accuracy(y_train_real, y_train_pred, lookup_step)
        y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], data["column_scaler"][test_var])
        valid_acc = get_accuracy(y_valid_real, y_valid_pred, lookup_step)
        y_test_real, y_test_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var])
        test_acc = get_accuracy(y_test_real, y_test_pred, lookup_step)
    else:
        y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], data["column_scaler"][test_var])
        valid_acc = get_accuracy(y_valid_real, y_valid_pred, lookup_step)
        train_acc = test_acc = 0

    return train_acc, valid_acc, test_acc 

def get_accuracy(y_real, y_pred, lookup_step):
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_real[:-lookup_step], y_pred[lookup_step:]))
    y_real = list(map(lambda current, future: int(float(future) > float(current)), y_real[:-lookup_step], y_real[lookup_step:]))

    return accuracy_score(y_real, y_pred)

def get_all_maes(model, test_tensorslice, valid_tensorslice, train_tensorslice, data):
    train_mae = get_mae(model, train_tensorslice, data)
    valid_mae = get_mae(model, valid_tensorslice, data)
    test_mae =  get_mae(model, test_tensorslice, data)

    return test_mae, valid_mae, train_mae

def get_mae(model, tensorslice, data):
    mse, mae = model.evaluate(tensorslice, verbose=0)
    mae = data["column_scaler"][test_var].inverse_transform([[mae]])[0][0]

    return mae

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

    symbol = ticker = "AGYS"

    time_s = time.time()
    data, train, valid, test = load_data(ticker, 300, True, True, 1, .2, ["open", "low", "high", "close", "mid", "volume", "stochas_fast_k", "stochas_fast_d"], 64, end_date=None)
    print("load data took " + str(time.time() - time_s))

    # time_s = time.time()
    # df = make_dataframe(ticker, limit=600, end_date=None)
    # print("make data took " + str(time.time() - time_s))

    # time_s = time.time()
    # api = get_api()
    # current_price = 0
    # barset = api.get_barset(symbol, "day", limit=1)
    # print("getting one day took " + str(time.time() - time_s))