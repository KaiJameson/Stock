from functions import silence_tensorflow, layer_name_converter
silence_tensorflow()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque
from api_key import (real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key,
intrinio_sandbox_key, intrinio_production_key)
from environ import (test_var, back_test_days, to_plot, test_money, stocks_traded, 
using_all_accuracies, directory_dict)
from time_functs import get_time_string,  get_trade_day_back, get_full_end_date, make_Timestamp
from io_functs import make_current_price, plot_graph, excel_output, write_nn_report
from error_functs import error_handler, net_error_handler
from symbols import trading_real_money
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from intrinio_sdk.rest import ApiException
import alpaca_trade_api as tradeapi
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
import xgboost as xgb
import intrinio_sdk as intrinio
import time
import datetime
import os
import sys


def nn_report(symbol, total_time, model, data, test_acc, valid_acc, train_acc, N_STEPS, classification):
    time_string = get_time_string()
    # predict the future price
    future_price = predict(model, data, N_STEPS)
    
    y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var], classification)

    report_dir = directory_dict["reports_directory"] + "/" + symbol + "/" + time_string + ".txt"
    
    if to_plot:
        plot_graph(y_real, y_pred, symbol, back_test_days, time_string)

    total_minutes = total_time / 60

    real_y_values = y_real[-back_test_days:]
    predicted_y_values = y_pred[-back_test_days:]

    curr_price = real_y_values[-1]
    percent = future_price / curr_price

    write_nn_report(symbol, report_dir, total_minutes, real_y_values, predicted_y_values,
        curr_price, future_price, test_acc, valid_acc, train_acc, y_real, y_pred)
    excel_output(symbol, curr_price, future_price)

    return percent, future_price

def get_values(items):
    data = {}
    for symbol, bar in items:
        open_values = []
        close_values = []
        low_values = []
        high_values = []
        volume = []
        times = []
        for day in bar:
            open_price = day.o
            close_price = day.c
            low_price = day.l
            high_price = day.h
            vol = day.v
            time = day.t
            open_values.append(open_price)
            close_values.append(close_price)
            low_values.append(low_price)
            high_values.append(high_price)
            volume.append(vol)
            times.append(time)
        data["open"] = open_values
        data["low"] = low_values
        data["high"] = high_values
        data["close"] = close_values
        data["volume"] = volume
        data["time"] = times
    df = pd.DataFrame(data=data)
    return df

def get_alpaca_data(symbol, end_date, api, timeframe="day", limit=1000):
    frames = []	    

    if end_date is not None:
        end_date = make_Timestamp(end_date + datetime.timedelta(1)) 

    while limit > 1000:
        if end_date is not None:
            other_barset = api.get_barset(symbols=symbol, timeframe="day", limit=1000, until=end_date)
            new_df = get_values(other_barset.items()) 
            limit -= 1000
            end_date = get_trade_day_back(end_date, 1000)
        else:
            other_barset = api.get_barset(symbols=symbol, timeframe="day", limit=1000, until=end_date)
            new_df = get_values(other_barset.items()) 
            limit -= 1000
            end_date = get_trade_day_back(get_full_end_date(), 1000)

        frames.insert(0, new_df)


    if limit > 0:	
        if end_date is not None:
            barset = api.get_barset(symbols=symbol, timeframe="day", limit=limit, until=end_date)
            items = barset.items()
            new_df = get_values(items)
        else:	 
            barset = api.get_barset(symbols=symbol, timeframe="day", limit=limit)
            items = barset.items() 	
            new_df = get_values(items)	

        frames.insert(0, new_df)

    df = pd.concat(frames) 
    df = alpaca_date_converter(df)  
    
    return df

def make_dataframe(symbol, feature_columns, limit=1000, end_date=None, to_print=True):
    api = get_api()

    df = get_alpaca_data(symbol, end_date, api, limit=limit)
    
    if "mid" in feature_columns:
        df["mid"] = (df.low + df.high) / 2

    if "volume" not in feature_columns:
        df = df.drop(columns=["volume"])

    if "S&P" in feature_columns:
        df2 = get_alpaca_data("SPY", end_date, api, limit=limit)
        df["S&P"] = df2.close 

    if "DOW" in feature_columns:
        df2 = get_alpaca_data("DIA", end_date, api, limit=limit)
        df["DOW"] = df2.close 

    if "NASDAQ" in feature_columns:
        df2 = get_alpaca_data("QQQ", end_date, api, limit=limit)
        df["NASDAQ"] = df2.close 
    
    if "VIX" in feature_columns:
        df2 = get_alpaca_data("VIXY", end_date, api, limit=limit)
        df["VIX"] = df2.close

    if "7MA" in feature_columns:
        df["7MA"] = df.close.rolling(window=7).mean()
    
    if ("upper_band" or "lower_band") in feature_columns:
        upperband, middleband, lowerband = ta.BBANDS(df.close, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
        df["upper_band"] = upperband
        df["lower_band"] = lowerband

    if "OBV" in feature_columns:
        df["OBV"] = ta.OBV(df.close, df.volume)

    if "relative_strength_index" in feature_columns:
        df["relative_strength_index"] = ta.RSI(df.close)
    
    if "lin_regres" in feature_columns:
        df["lin_regres"] = ta.LINEARREG(df.close, timeperiod=14)

    if "lin_regres_angle" in feature_columns:
        df["lin_regres_angle"] = ta.LINEARREG_ANGLE(df.close, timeperiod=14)

    if "lin_regres_intercept" in feature_columns:
        df["lin_regres_intercept"] = ta.LINEARREG_INTERCEPT(df.close, timeperiod=14)

    if "lin_regres_slope" in feature_columns:
        df["lin_regres_slope"] = ta.LINEARREG_SLOPE(df.close, timeperiod=14)

    if "pearsons_correl" in feature_columns:
        df["pearsons_correl"] = ta.CORREL(df.high, df.low, timeperiod=30)

    if "money_flow_ind" in feature_columns:
        df["money_flow_ind"] = ta.MFI(df.high, df.low, df.close, df.volume, timeperiod=14)

    if "wills_r" in feature_columns:
        df["wills_r"] = ta.WILLR(df.high, df.low, df.close, timeperiod=14)

    if "std_dev" in feature_columns:
        df["std_dev"] = ta.STDDEV(df.close, timeperiod=5, nbdev=1)

    if ("min" or "max") in feature_columns:
        minimum, maximum = ta.MINMAX(df.close, timeperiod=30)
        df["min"] = minimum
        df["max"] = maximum

    if "commodity_channel_ind" in feature_columns:
        df["commodity_channel_ind"] = ta.CCI(df.high, df.low, df.close, timeperiod=14)

    if "parabolic_SAR" in feature_columns:
        df["parabolic_SAR"] = ta.SAR(df.high, df.low)

    if "parabolic_SAR_extended" in feature_columns:
        df["parabolic_SAR_extended"] = ta.SAREXT(df.high, df.low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, 
        accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    if "rate_of_change" in feature_columns:
        df["rate_of_change"] = ta.ROC(df.close, timeperiod=10)

    if "ht_dcperiod" in feature_columns:
        df["ht_dcperiod"] = ta.HT_DCPERIOD(df.close)

    if "ht_trendmode" in feature_columns:
        df["ht_trendmode"] = ta.HT_TRENDMODE(df.close)

    if "ht_dcphase" in feature_columns:
        df["ht_dcphase"] = ta.HT_DCPHASE(df.close)

    if ("ht_inphase" or "quadrature") in feature_columns:
        df["ht_inphase"], df["quadrature"] = ta.HT_PHASOR(df.close)

    if ("ht_sine" or "ht_leadsine") in feature_columns:
        df["ht_sine"], df["ht_leadsine"] = ta.HT_SINE(df.close)

    if "ht_trendline" in feature_columns:
        df["ht_trendline"] = ta.HT_TRENDLINE(df.close)

    if "momentum" in feature_columns:
        df["momentum"] = ta.MOM(df.close, timeperiod=10)

    if "abs_price_osc" in feature_columns:
        df["abs_price_osc"] = ta.APO(df.close, fastperiod=12, slowperiod=26, matype=0)

    if "KAMA" in feature_columns:
        df["KAMA"] = ta.KAMA(df.close, timeperiod=30)

    if "typical_price" in feature_columns:    
        df["typical_price"] = ta.TYPPRICE(df.high, df.low, df.close)

    if "ultimate_osc" in feature_columns:
        df["ultimate_osc"] = ta.ULTOSC(df.high, df.low, df.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    if "chaikin_line" in feature_columns:
        df["chaikin_line"] = ta.AD(df.high, df.low, df.close, df.volume)

    if "chaikin_osc" in feature_columns:
        df["chaikin_osc"] = ta.ADOSC(df.high, df.low, df.close, df.volume, fastperiod=3, slowperiod=10)

    if "norm_average_true_range" in feature_columns:
        df["norm_average_true_range"] = ta.NATR(df.high, df.low, df.close, timeperiod=14)

    if "median_price" in feature_columns:
        df["median_price"] = ta.MEDPRICE(df.high, df.low)

    if "variance" in feature_columns:
        df["variance"] = ta.VAR(df.close, timeperiod=5, nbdev=1)

    if ("aroon_down" or "aroon_up") in feature_columns:
        df["aroon_down"], df["aroon_up"] = ta.AROON(df.high, df.low, timeperiod=14)

    if "aroon_osc" in feature_columns:
        df["aroon_osc"] = ta.AROONOSC(df.high, df.low, timeperiod=14)

    if "balance_of_pow" in feature_columns:
        df["balance_of_pow"] = ta.BOP(df.open, df.high, df.low, df.close)

    if "chande_momen_osc" in feature_columns:    
        df["chande_momen_osc"] = ta.CMO(df.close, timeperiod=14)

    if ("macd" or "macdsignal" or "macdhist") in feature_columns:
        df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)

    if ("control_MACD" or "control_MACD_signal" or "control_MACD_hist") in feature_columns:
        df["control_MACD"], df["control_MACD_signal"], df["control_MACD_hist"] = ta.MACDEXT(df.close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

    if ("fix_MACD" or "fix_MACD_signal" or "fix_MACD_hist") in feature_columns:
        df["fix_MACD"], df["fix_MACD_signal"], df["fix_MACD_hist"] = ta.MACDFIX(df.close, signalperiod=9)

    if "minus_directional_ind" in feature_columns:
        df["minus_directional_ind"] = ta.MINUS_DI(df.high, df.low, df.close, timeperiod=14)

    if "minus_directional_move" in feature_columns:
        df["minus_directional_move"] = ta.MINUS_DM(df.high, df.low, timeperiod=14)

    if "plus_directional_ind" in feature_columns:
        df["plus_directional_ind"] = ta.PLUS_DI(df.high, df.low, df.close, timeperiod=14)

    if "plus_directional_move" in feature_columns:
        df["plus_directional_move"] = ta.PLUS_DM(df.high, df.low, timeperiod=14)

    if "percentage_price_osc" in feature_columns:
        df["percentage_price_osc"] = ta.PPO(df.close, fastperiod=12, slowperiod=26, matype=0)

    if ("stochas_fast_k" or "stochas_fast_d") in feature_columns:
        df["stochas_fast_k"], df["stochas_fast_d"] = ta.STOCHF(df.high, df.low, df.close, fastk_period=5, fastd_period=3, fastd_matype=0)

    if ("stochas_relative_strength_k" or "stochas_relative_strength_d") in feature_columns:
        df["stochas_relative_strength_k"], df["stochas_relative_strength_d"] = ta.STOCHRSI(df.close, fastk_period=5, fastd_period=3, fastd_matype=0)

    if ("stochas_slowk" or "stochas_slowd") in feature_columns:
        df["stochas_slowk"], df["stochas_slowd"] = ta.STOCH(df.high, df.low, df.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    if "TRIX" in feature_columns:
        df["TRIX"] = ta.TRIX(df.close, timeperiod=30)

    if "weighted_moving_avg" in feature_columns:
        df["weighted_moving_avg"] = ta.WMA(df.close, timeperiod=30)

    if ("upband" or "midband" or "lowband") in feature_columns:
        df["upband"], df["midband"], df["lowband"] = ta.BBANDS(df.close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

    if "double_exponetial_moving_avg" in feature_columns:
        df["double_exponetial_moving_avg"] = ta.DEMA(df.close, timeperiod=30)

    if "exponential_moving_avg" in feature_columns:
        df["exponential_moving_avg"] = ta.EMA(df.close, timeperiod=30)

    if ("MESA_mama" or "MESA_fama") in feature_columns:
        df["MESA_mama"], df["MESA_fama"] = ta.MAMA(df.close)

    if "midpoint" in feature_columns:
        df["midpoint"] = ta.MIDPOINT(df.close, timeperiod=14)

    if "midprice" in feature_columns:
        df["midprice"] = ta.MIDPRICE(df.high, df.low, timeperiod=14)

    if "triple_exponential_moving_avg" in feature_columns:
        df["triple_exponential_moving_avg"] = ta.TEMA(df.close, timeperiod=30)

    if "triangular_moving_avg" in feature_columns:
        df["triangular_moving_avg"] = ta.TRIMA(df.close, timeperiod=30)

    if "avg_directional_movement_index" in feature_columns:
        df["avg_directional_movement_index"] = ta.ADX(df.high, df.low, df.close, timeperiod=14)

    if "true_range" in feature_columns:
        df["true_range"] = ta.TRANGE(df.high, df.low, df.close)

    if "avg_price" in feature_columns:
        df["avg_price"] = ta.AVGPRICE(df.open, df.high, df.low, df.close)

    if "weighted_close_price" in feature_columns:
        df["weighted_close_price"] = ta.WCLPRICE(df.high, df.low, df.close)

    if "beta" in feature_columns:
        df["beta"] = ta.BETA(df.high, df.low, timeperiod=5)

    if "time_series_for" in feature_columns:
        df["time_series_for"] = ta.TSF(df.close, timeperiod=14)

    if "day_of_week" in feature_columns:
        df = convert_date_values(df)

    # get_feature_importance(df)

    # sentiment_data(df)

    if to_print:
        pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_rows", None)
        print(df.head(1))
        print(df.tail(1))
        # print(df)
    return df

def alpaca_date_converter(df):
    df.index = df["time"]
    df = df.drop("time", axis=1)
    return df

def convert_date_values(df):	    
    df["day_of_week"] = df.index	
    df["day_of_week"] = df["day_of_week"].dt.dayofweek

    return df

def load_data(symbol, params, end_date=None, shuffle=True, scale=True, to_print=True):

    if to_print:
        print("Included features: " + str(params["FEATURE_COLUMNS"]))
    no_connection = True
    while no_connection:
        try:
            if end_date is not None:
                df = make_dataframe(symbol, params["FEATURE_COLUMNS"], params["LIMIT"], end_date, to_print)
            else:
                df = make_dataframe(symbol, params["FEATURE_COLUMNS"], params["LIMIT"], to_print=to_print)

            no_connection = False

        except Exception:
            net_error_handler(symbol, Exception)

    result = {}
    result["df"] = df.copy()

    for col in params["FEATURE_COLUMNS"]:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    
    # print("future: " + str(df["future"]))
    # print("future: " + str(df["future"][0]))
    # print("future: " + str(df["future"].values))
    # if params["LOSS"] == "huber_loss":
    #             print(df["future"].values)
    #             df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1), df["future"])

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in params["FEATURE_COLUMNS"]:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        result["column_scaler"] = column_scaler

    if params["LOSS"] == "huber_loss":
        df["future"] = df[test_var].shift(-params["LOOKUP_STEP"])
    else:
        df["future"] = df[test_var].shift(-params["LOOKUP_STEP"])
        df["future"] = list(map(lambda current, future: int(float(future) > float(current)), df.close, df.future))


    # add the target column (label) by shifting by `lookup_step`\
    
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[params["FEATURE_COLUMNS"]].tail(params["LOOKUP_STEP"]))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=params["N_STEPS"])
    for entry, target in zip(df[params["FEATURE_COLUMNS"]].values, df["future"].values):
        sequences.append(entry)
        if len(sequences) == params["N_STEPS"]:
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

    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    # print(X)
    # print(y)
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    
    # split the dataset
    result["X_train"], result["X_valid"], result["y_train"], result["y_valid"] = train_test_split(X, y, test_size=params["TEST_SIZE"], shuffle=shuffle)
    result["X_valid"], result["X_test"], result["y_valid"], result["y_test"] = train_test_split(result["X_valid"], result["y_valid"], test_size=.006, shuffle=shuffle)

    # print("in the load thingy " + str(result["X_valid"]))

    train = Dataset.from_tensor_slices((result["X_train"], result["y_train"]))
    valid = Dataset.from_tensor_slices((result["X_valid"], result["y_valid"]))
    test = Dataset.from_tensor_slices((result["X_test"], result["y_test"]))
    
    train = train.batch(params["BATCH_SIZE"])
    valid = valid.batch(params["BATCH_SIZE"])
    test = test.batch(params["BATCH_SIZE"])
    
    train = train.cache()
    valid = valid.cache()
    test = test.cache()

    train = train.prefetch(buffer_size=AUTOTUNE)
    valid = valid.prefetch(buffer_size=AUTOTUNE)
    test = test.prefetch(buffer_size=AUTOTUNE)

    # return the result
    return result, train, valid, test

def create_model(params):
    model = Sequential()
    bi_string = "Bidirectional" if params["BIDIRECTIONAL"] else ""
    # print(bi_string)
    for layer in range(len(params["LAYERS"])):
        if layer == 0:
            model_first_layer(model, params["LAYERS"], layer, params["N_STEPS"])
        elif layer == len(params["LAYERS"]) - 1:
            model_last_layer(model, params["LAYERS"], layer)
        else:
            model_hidden_layers(model, params["LAYERS"], layer)
    
        model.add(Dropout(params["DROPOUT"]))
    model.add(Dense(1, activation="linear"))
    if params["LOSS"] == "huber_loss":
        model.compile(loss=params["LOSS"], metrics=["mean_absolute_error"], optimizer=params["OPTIMIZER"])
    else:
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=params["OPTIMIZER"])
    # print(model.summary())
    return model

def model_first_layer(model, layers, ind, n_steps):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if layer_name == "Dense":
        print("You need to have a recurrent layer leading your model")
        print("otherwise everything breaks, limitation of the loading code.")
        print("Sorry buddy")
        sys.exit(-1)

    if (next_layer_name == "LSTM" or next_layer_name == "SRNN" or next_layer_name == "GRU"):
        model.add(layers[ind][1](layers[ind][0], return_sequences=True, input_shape=(None, n_steps)))
    else:
        model.add(layers[ind][1](layers[ind][0], return_sequences=False, input_shape=(None, n_steps)))

    return model

def model_hidden_layers(model, layers, ind):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if (not(layer_name == "LSTM" or layer_name == "SRNN" or layer_name == "GRU")):
        model.add(layers[ind][1](layers[ind][0]))
    else:
        if (next_layer_name == "LSTM" or next_layer_name == "SRNN" or next_layer_name == "GRU"):
            model.add(layers[ind][1](layers[ind][0], return_sequences=True))
        else:
            model.add(layers[ind][1](layers[ind][0], return_sequences=False))

    return model

def model_last_layer(model, layers, ind):
    layer_name = layer_name_converter(layers[ind])

    if (not(layer_name == "LSTM" or layer_name == "SRNN" or layer_name == "GRU")):
        model.add(layers[ind][1](layers[ind][0]))
    else:
        model.add(layers[ind][1](layers[ind][0], return_sequences=False))
    
    return model

def load_model_with_data(symbol, current_date, params, directory, model_name):
    data, train, valid, test = load_data(symbol, params, current_date, shuffle=False, to_print=False)
    model = create_model(params)
    model.load_weights(directory + "/" + params["SAVE_FOLDER"] + "/" + model_name + ".h5")

    return data, model

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
    if not classification:
        predicted_val = column_scaler[test_var].inverse_transform(prediction)[0][0]
    else:
        predicted_val = prediction[0][0]
    return predicted_val

def getOwnedStocks():
    api = get_api()
    positions = api.list_positions()
    owned = {}
    for position in positions:
        owned[position.symbol] = position.qty
    return owned

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
                    qty = owned.pop(symbol)

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

        except Exception:
            error_handler(symbol, Exception)

            
    print("The Owned list: " + str(owned))
    print("The buy list: " + str(buy_list))

    account_equity = float(api.get_account().equity)
    buy_power = float(api.get_account().cash)

    value_in_stocks = 1 - (buy_power / account_equity)

    print("Value in stocks: " + str(value_in_stocks))
    print("Account equity: " + str(account_equity))

    stock_portion_adjuster = 0

    if value_in_stocks > .7:
        stock_portion_adjuster = len(buy_list)
    elif value_in_stocks > .3:
        if (len(buy_list) / stocks_traded) > .8:
            stock_portion_adjuster = len(buy_list) # want 100%
        elif (len(buy_list) / stocks_traded) > .6:
            stock_portion_adjuster = len(buy_list)  # want 100%
        elif (len(buy_list) / stocks_traded) > .4:
            stock_portion_adjuster = len(buy_list) / .90 # want 90%
        else:
            stock_portion_adjuster = len(buy_list) / .70 # want 70%
    else:
        if (len(buy_list) / stocks_traded) > .8:
            stock_portion_adjuster = len(buy_list) # want 100%
        elif (len(buy_list) / stocks_traded) > .6:
            stock_portion_adjuster = len(buy_list) / .90 # want 90%
        elif (len(buy_list) / stocks_traded) > .4:
            stock_portion_adjuster = len(buy_list) / .70 # want 70%
        else:
            stock_portion_adjuster = len(buy_list) / .60 # want 60%
            

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
                
        except Exception:
            error_handler(symbol, Exception)

def get_api():
    if trading_real_money:
        api = tradeapi.REST(real_api_key_id, real_api_secret_key, base_url="https://api.alpaca.markets")
    else:
        api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")

    return api

def sentiment_data(df):
    finviz_url = "https://finviz.com/quote.ashx?t="

    # nltk.download('vader_lexicon')

    time_s = time.time()

    news_tables = {}
    tickers = ["AGYS", "BG"]

    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'}) 
        response = urlopen(req)    
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response, features="lxml")
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[ticker] = news_table

    # Read one single day of headlines for 'AMZN' 
    amzn = news_tables['AGYS']
    # Get all the table rows tagged in HTML with <tr> into 'amzn_tr'
    amzn_tr = amzn.findAll('tr')

    # for i, table_row in enumerate(amzn_tr):
    #     # Read the text of the element 'a' into 'link_text'
    #     a_text = table_row.a.text
    #     # Read the text of the element 'td' into 'data_text'
    #     td_text = table_row.td.text
    #     # Print the contents of 'link_text' and 'data_text' 
    #     print(a_text)
    #     print(td_text)
    #     # Exit after printing 4 rows of data
    #     # if i == 3:
    #     #     break


    parsed_news = []

    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text() 
            # splice text in the td tag into a list 
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                the_time = date_scrape[0]
                
            # else load 'date' as the 1st element and 'time' as the second    
            else:
                date = date_scrape[0]
                the_time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'  
            ticker = file_name.split('_')[0]
            
            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, date, the_time, text])
            
    parsed_news

    vader = SentimentIntensityAnalyzer()

    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']

    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

    print(parsed_and_scored_news)

    plt.rcParams['figure.figsize'] = [10, 6]

    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.groupby(['ticker','date']).mean()

    # Unstack the column ticker
    mean_scores = mean_scores.unstack()

    # Get the cross-section of compound in the 'columns' axis
    mean_scores = mean_scores.xs('compound', axis="columns").transpose()

    # Plot a bar chart with pandas
    mean_scores.plot(kind = 'bar')
    plt.grid()


    print("this took " + str(time.time() - time_s))


def intrinio_news():
    intrinio.ApiClient().set_api_key(intrinio_sandbox_key)
    intrinio.ApiClient().allow_retries(True)

    identifier = "AXP"
    page_size = 1250
    next_page = ""

    response = intrinio.CompanyApi().get_company_news(identifier, page_size=page_size, next_page=next_page)
    print(str(response))

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
    
def get_all_accuracies(model, data, lookup_step, classification=False):
    if using_all_accuracies:
        y_train_real, y_train_pred = return_real_predict(model, data["X_train"], data["y_train"], 
        data["column_scaler"][test_var], classification)
        train_acc = get_accuracy(y_train_real, y_train_pred, lookup_step)
        y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], 
        data["column_scaler"][test_var], classification)
        valid_acc = get_accuracy(y_valid_real, y_valid_pred, lookup_step)
        y_test_real, y_test_pred = return_real_predict(model, data["X_test"], data["y_test"],
         data["column_scaler"][test_var], classification)
        test_acc = get_accuracy(y_test_real, y_test_pred, lookup_step)
    else:
        # print("data X_valid" + str(data["X_valid"]))
        y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], 
        data["column_scaler"][test_var], classification)
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

def return_real_predict(model, X_data, y_data, column_scaler, classification=False):
    y_pred = model.predict(X_data)
    y_real = np.squeeze(column_scaler.inverse_transform(np.expand_dims(y_data, axis=0)))
    if not classification:
        y_pred = np.squeeze(column_scaler.inverse_transform(y_pred))

    return y_real, y_pred

if __name__ == "__main__":
    from functions import get_test_name
    from paca_model import saveload_neural_net

    defaults = {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.4,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 200,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "tuning4"
    }

    

    symbol = "AGYS"
    saveload_neural_net(symbol, params=defaults)
    
    start_time = time.time()
    model_name = (symbol + "-" + get_test_name(defaults))

    print("\n~~~Now Starting " + symbol + "~~~")
    
    time_s = time.time()
    data, train, valid, test = load_data(symbol, defaults, shuffle=False, to_print=False)
    print("Loading the data took " + str(time.time() - time_s) + " seconds")    

    time_s = time.time()
    model = create_model(defaults)
    model.load_weights(directory_dict["model_directory"] + "/" + defaults["SAVE_FOLDER"] + "/" + model_name + ".h5")
    print("Loading the model took " + str(time.time() - time_s) + " seconds")    

    time_s = time.time()
    train_acc, valid_acc, test_acc = get_all_accuracies(model, data, defaults["LOOKUP_STEP"], False)
    print("Getting the accuracies took " + str(time.time() - time_s) + " seconds")   

    total_time = time.time() - start_time
    time_s = time.time()
    percent, future_price = nn_report(symbol, total_time, model, data, test_acc, valid_acc, 
    train_acc, defaults["N_STEPS"], False)
    y_real, y_pred = return_real_predict(model, data["X_valid"], data["y_valid"], data["column_scaler"][test_var], True)
    print(f"real: {y_real}")
    print(f"predict: {y_pred}")
    future_price = predict(model, data, defaults["N_STEPS"], False) 
    print("NN report took " + str(time.time() - time_s) + " seconds")

    print(f"predicted value: {future_price}")


