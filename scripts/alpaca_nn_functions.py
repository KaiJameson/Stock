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
from functions import deleteFiles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
import datetime
import math 
import talib as ta


def nn_report(ticker, total_time, model, data, test_acc, valid_acc, train_acc, test_mae, valid_mae, 
train_mae, N_STEPS):
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
    f.write("The test mean absolute error is: " + str(round(test_mae, 4)) + "\n")
    f.write("The validation mean absolute error is: " + str(round(valid_mae, 4)) + "\n")
    f.write("The training absolute error is: " + str(round(train_mae, 4)) + "\n")
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

    excel_output(ticker, curr_price, future_price)

    return percent

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

    # df["simple_rolling_avg"] = df.close.rolling(window=10).mean()
    
    # upperband, middleband, lowerband = ta.BBANDS(df.close, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
    # df["upper_band"] = upperband
    # df["lower_band"] = lowerband

    # df["OBV"] = ta.OBV(df.close, df.volume)

    # df["relative_strength_index"] = ta.RSI(df.close)

    # df["linear_regression"] = ta.LINEARREG(df.close, timeperiod=14)

    # df["linear_regression_angle"] = ta.LINEARRG_ANGLE(df.close, timeperiod=14)

    # df["linear_regression_intercept"] = ta.LINEARREG_INTERCEPT(df.close, timeperiod=14)

    # df["linear_regression_slope"] = ta.LINEARREG_SLOPE(df.close, timeperiod=14)

    # df["BETA"] = ta.BETA(df.high, df.low, timeperiod=5)

    # df["CORRELATION"] = ta.CORREL(df.high, df.low, timeperiod=30)

    # df["money_flow_index"] = ta.MFI(df.high, df.low, df.close, df.volume, timeperiod=14)

    # df["williams_r"] = ta.WILLR(df.high, df.low, df.close, timeperiod=14)

    # df["standard_deviation"] = ta.STDDEV(df.close, timeperiod=5, nbdev=1)

    # minimum, maximum = ta.MINMAX(df.close, timeperiod=30)
    # df["minimum"] = minimum
    # df["maximum"] = maximum
 
    # df["time_series_forecast"] = ta.TSF(df.close, timeperiod=14)

    # df["commodity_channel_index"] = ta.CCI(df.high, df.low, df.close, timeperiod=14)

    # df["average_true_range"] = ta.ATR(df.high, df.low, df.close, timeperiod=14)

    # df["average_directional_movement_index"] = ta.ADX(df.high, df.low, df.close, timeperiod=14)

    # df["parabolic_SAR"] = ta.SAR(df.high, df.low)

    # df["parabolic_SAR_extended"] = ta.SAREXT(df.high, df.low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, 
    # accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    # df["MACD"], df["MACD_signal"], df["MACD_hist"] = ta.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)

    # df["rate_of_change"] = ta.ROC(df.close, timeperiod=10)

    # df["ht_trendmode"] = ta.HT_TRENDMODE(df.close)

    # df["ht_dcphase"] = ta.HT_DCPHASE(df.close)

    # df["ht_inphase"], df["quadrature"] = ta.HT_PHASOR(df.close)

    # df["ht_sine"], df["ht_leadsine"] = ta.HT_SINE(df.close)

    # df["ht_trendline"] = ta.HT_TRENDLINE(df.close)

    # df["momentum"] = ta.MOM(df.close, timeperiod=10)

    # df["absolute_price_oscillator"] = ta.APO(df.close, fastperiod=12, slowperiod=26, matype=0)

    # df["average_true_range"] = ta.ATR(df.high, df.low, df.close, timeperiod=14)

    # df["KAMA"] = ta.KAMA(df.close, timeperiod=30)

    # df["typical_price"] = ta.TYPPRICE(df.high, df.low, df.close)

    # df["ultimate_oscillator"] = ta.ULTOSC(df.high, df.low, df.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # df["chaikin_line"] = ta.AD(df.high, df.low, df.close, df.volume)

    # df["chaikin_oscillator"] = ta.ADOSC(df.high, df.low, df.close, df.volume, fastperiod=3, slowperiod=10)

    # df["normalized_average_true_range"] = ta.NATR(df.high, df.low, df.close, timeperiod=14)

    # df["median_price"] = ta.MEDPRICE(df.high, df.low)

    # df["variance"] = ta.VAR(df.close, timeperiod=5, nbdev=1)

    # df["aroon_down"], df["aroon_up"] = ta.AROON(df.high, df.low, timeperiod=14)

    # df["aroon_oscillator"] = ta.AROONOSC(df.high, df.low, timeperiod=14)

    # df["balance_of_power"] = ta.BOP(df.open, df.high, df.low, df.close)

    # df["chande_momentum_oscillator"] = ta.CMO(df.close, timeperiod=14)

    # df["control_MACD"], df["control_MACD_signal"], df["control_MACD_hist"] = ta.MACEXT(df.close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

    # df["fixed_MACD"], df["fixed_MACD_signal"], df["fixed_MACD_hist"] = ta.MACEXT(df.close, signalperiod=9)

    # df["minus_directional_indicator"] = ta.MINUS_DI(df.high, df.low, df.close, timeperiod=14)

    # df["minus_directional_movement"] = ta.MINUS_DM(df.high, df.low, timeperiod=14)

    # df["plus_directional_indicator"] = ta.PLUS_DI(df.high, df.low, df.close, timeperiod=14)

    # df["plus_directional_movement"] = ta.PLUS_DM(df.high, df.low, timeperiod=14)

    # df["percentage_price_oscillator"] = ta.PPO(df.close, fastperiod=12, slowperiod=26, matype=0)

    # df["stochastic_fast_k"], df["stochastic_fast_d"] = ta.STOCHF(df.high, df.low, fastk_period=5, fastd_period=3, fastd_matype=0)

    # df["stochastic_relative_strength_k"] = df["stochastic_relative_strength_d"] = ta.STOCHRSI(df.close, fastk_period=5, fastd_period=3, fastd_matype=0)

    # df["TRIX"] = ta.TRIX(df.close, timeperiod=30)

    # df["weighted_moving_average"] = ta.WMA(df.close, timeperiod=30)

    # TODO figure out what periods to use here
    # df["moving_average_with_variable_period"] = ta.MAVP(df.close, periods, minperiod=2, maxperiod=30, matype=0)

    # Group 1
    # df["two_crows"] = ta.CDL2CROWS(df.open, df.high, df.low, df.close)
    # df["three_black_crows"] = ta.CDL3BLACKCROWS(df.open, df.high, df.low, df.close)
    # df["three_inside_updown"] = ta.CDL3INSIDE(df.open, df.high, df.low, df.close)
    # df["three_line_strike"] = ta.CDL3LINESTRIKE(df.open, df.high, df.low, df.close)
    # df["three_outside_updown"] = ta.CDL3OUTSIDE(df.open, df.high, df.low, df.close)

    # Group 2
    # df["three_stars_in_the_south"] = ta.CDL3STARSINSOUTH(df.open, df.high, df.low, df.close)
    # df["three_advancing_white_soldiers"] = ta.CDL3WHITESOLDIERS(df.open, df.high, df.low, df.close)
    # df["abandoned_baby"] = ta.CDLABANDONEDBABY(df.open, df.high, df.low, df.close)
    # df["advance_block"] = ta.CDLADVANCEBLOCK(df.open, df.high, df.low, df.close)
    # df["belt_hold"] = ta.CDLBELTHOLD(df.open, df.high, df.low, df.close)

    # Group 3
    # df["breakaway"] = ta.CDLBREAKAWAY(df.open, df.high, df.low, df.close)
    # df["closing_marubozu"] = ta.CDLCLOSINGMARUBOZU(df.open, df.high, df.low, df.close)
    # df["concealing_baby_swallow"] = ta.CDLCONCEALBABYSWALL(df.open, df.high, df.low, df.close)
    # df["counterattack"] = ta.CDLCOUNTERATTACK(df.open, df.high, df.low, df.close)
    # df["dark_cloud_cover"] = ta.CDLDARKCLOUDCOVER(df.open, df.high, df.low, df.close)

    # Group 4
    # df["doji"] = ta.CDLDOJI(df.open, df.high, df.low, df.close)
    # df["doji_star"] = ta.CDLDOJISTAR(df.open, df.high, df.low, df.close)
    # df["dragonfly_doji"] = ta.CDLDRAGONFLYDOJI(df.open, df.high, df.low, df.close)
    # df["engulfing_pattern"] = ta.CDLENGULFING(df.open, df.high, df.low, df.close)
    # df["evening_doji_star"] = ta.CDLEVENINGDOJISTAR(df.open, df.high, df.low, df.close)

    # Group 5
    # df["evening_star"] = ta.CDLEVENINGSTAR(df.open, df.high, df.low, df.close)
    # df["updown_gap_sidebyside_white_lines"] = ta.CDLGAPSIDESIDEWHITE(df.open, df.high, df.low, df.close)
    # df["gravestone_doji"] = ta.CDLGRAVESTONEDOJI(df.open, df.high, df.low, df.close)
    # df["hammer"] = ta.CDLHAMMER(df.open, df.high, df.low, df.close)
    # df["hanging_man"] = ta.CDLHANGINGMAN(df.open, df.high, df.low, df.close)

    # Group 6
    # df["harami_pattern"] = ta.CDLHARAMI(df.open, df.high, df.low, df.close)
    # df["harami_cross_pattern"] = ta.CDLHARAMICROSS(df.open, df.high, df.low, df.close)
    # df["high_wave_candle"] = ta.CDLHIGHWAVE(df.open, df.high, df.low, df.close)
    # df["hikkake_pattern"]= ta.CDLHIKKAKE(df.open, df.high, df.low, df.close)
    # df["modified_hikkake_pattern"]= ta.CDLHIKKAKEMOD(df.open, df.high, df.low, df.close)

    # Group 7
    # df["homing_pigeon"] = ta.CDLHOMINGPIGEON(df.open, df.high, df.low, df.close)
    # df["identical_three_crows"] = ta.CDLIDENTICAL3CROWS(df.open, df.high, df.low, df.close)
    # df["in_neck_pattern"] = ta.CDLINNECK(df.open, df.high, df.low, df.close)
    # df["inverted_hammer"] = ta.CDLINVERTEDHAMMER(df.open, df.high, df.low, df.close)
    # df["kicking"] = ta.CDLKICKING(df.open, df.high, df.low, df.close)

    # Group 8
    # df["kicking_bull/bear_determined_by_longer_marubozu"] = ta.CDLKICKINGBYLENGTH(df.open, df.high, df.low, df.close)
    # df["ladder_bottom"] = ta.CDLLADDERBOTTOM(df.open, df.high, df.low, df.close)
    # df["long_legged_doji"] = ta.CDLLONGLEGGEDDOJI(df.open, df.high, df.low, df.close)
    # df["long_line_candle"] = ta.CDLLONGLINE(df.open, df.high, df.low, df.close)
    # df["marubozu"] = ta.CDLMARUBOZU(df.open, df.high, df.low, df.close)

    # Group 9
    # df["matching_low"] = ta.CDLMATCHINGLOW(df.open, df.high, df.low, df.close)
    # df["mat_hold"] = ta.CDLMATHOLD(df.open, df.high, df.low, df.close)
    # df["morning_doji_star"] = ta.CDLMORNINGDOJISTAR(df.open, df.high, df.low, df.close)
    # df["morning_star"] = ta.CDLMORNINGSTAR(df.open, df.high, df.low, df.close)
    # df["on_neck_pattern"] = ta.CDLONNECK(df.open, df.high, df.low, df.close)

    # Group 10
    # df["piercing_pattern"] = ta.CDLPIERCING(df.open, df.high, df.low, df.close)
    # df["rickshaw_man"] = ta.CDLRICKSHAWMAN(df.open, df.high, df.low, df.close)
    # df["rising_falling_three_methods"] = ta.CDLRISEFALL3METHODS(df.open, df.high, df.low, df.close)
    # df["separating_lines"] = ta.CDLSEPARATINGLINES(df.open, df.high, df.low, df.close)
    # df["shooting_star"] = ta.CDLSHOOTINGSTAR(df.open, df.high, df.low, df.close)

    # Group 11
    # df["short_line_candle"] = ta.CDLSHORTLINE(df.open, df.high, df.low, df.close)
    # df["spinning_top"] = ta.CDLSPINNINGTOP(df.open, df.high, df.low, df.close)
    # df["stalled_pattern"] = ta.CDLSTALLEDPATTERN(df.open, df.high, df.low, df.close)
    # df["stick_sandwich"] = ta.CDLSTICKSANDWICH(df.open, df.high, df.low, df.close)
    # df["takuri"] = ta.CDLTAKURI(df.open, df.high, df.low, df.close)

    # Group 12
    # df["tasuki_gap"] = CDLTASUKIGAP(df.open, df.high, df.low, df.close)  
    # df["thrusting_pattern"] = CDLTHRUSTING(df.open, df.high, df.low, df.close)
    # df["tristar_pattern"] = CDLTRISTAR(df.open, df.high, df.low, df.close)
    # df["unique_3_river"] = CDLUNIQUE3RIVER(df.open, df.high, df.low, df.close)
    # df["upside_gap_two_crows"] = CDLUPSIDEGAP2CROWS(df.open, df.high, df.low, df.close)
    # df["upside/downside_gap_three_methods"]= CDLXSIDEGAP3METHODS(df.open, df.high, df.low, df.close)

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df.head(100))
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

# , "rolling_avg""SAR" "KAMA" "williams_r" 
def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, test_size=0.2, 
feature_columns=["open", "low", "high", "close", "mid", "volume", "ht_trendmode"],
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
        if accuracy >= .55:
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
    
    real_price = ([ 110, 100, 110, 100, 110, 100,])
    predict =([ 110, 100, 110, 100, 110, 100])
   
    money = 100
    print(str(model_money(money, real_price, predict)))
    print(str(perfect_money(money, real_price)))

