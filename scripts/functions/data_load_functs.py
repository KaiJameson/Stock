from functions.time_functs import make_Timestamp, get_trade_day_back, get_full_end_date
from functions.error_functs import net_error_handler
from functions.trade_functs import get_api
from functions.time_functs import modify_timestamp
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter, cwt
from collections import deque
import talib as ta
import pandas as pd
import numpy as np
import datetime
import time


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
        data["o"] = open_values
        data["l"] = low_values
        data["h"] = high_values
        data["c"] = close_values
        data["v"] = volume
        data["time"] = times
    df = pd.DataFrame(data=data)
    return df

def get_alpaca_data(symbol, end_date, api, timeframe="day", limit=1000):
    frames = []	    

    # print(f"full {get_full_end_date()}")

    if end_date is not None:
        end_date = make_Timestamp(end_date + datetime.timedelta(1)) 

    while limit > 1000:
        if end_date is not None:
            other_barset = api.get_barset(symbols=symbol, timeframe="day", limit=1000, until=end_date)
            new_df = get_values(other_barset.items()) 
            limit -= 1000
            end_date = modify_timestamp(1, end_date)
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
    df = df.drop_duplicates()
    df = alpaca_date_converter(df)  
    # print(df)    
    return df

def make_dataframe(symbol, feature_columns, limit=1000, end_date=None, to_print=True):
    api = get_api()
    no_connection = True
    while no_connection:
        try:
            df = get_alpaca_data(symbol, end_date, api, limit=limit)
            no_connection = False

        except Exception:
            net_error_handler(symbol, Exception)

    df["m"] = (df.l + df.h) / 2

    if "sc" in feature_columns:
        df["sc"] = savgol_filter(df.c, 7, 3)

    if "so" in feature_columns:
        df["so"] = savgol_filter(df.o, 7, 3)

    if "sl" in feature_columns:
        df["sl"] = savgol_filter(df.l, 7, 3)

    if "sh" in feature_columns:
        df["sh"] = savgol_filter(df.h, 7, 3)
    
    if "sm" in feature_columns:
        df["sm"] = savgol_filter(df.m, 7, 3)

    if "sv" in feature_columns:
        df["sv"] = savgol_filter(df.v, 7, 3)

    if "S&P" in feature_columns:
        df2 = get_alpaca_data("SPY", end_date, api, limit=limit)
        df["S&P"] = df2.c 

    if "DOW" in feature_columns:
        df2 = get_alpaca_data("DIA", end_date, api, limit=limit)
        df["DOW"] = df2.c 

    if "NASDAQ" in feature_columns:
        df2 = get_alpaca_data("QQQ", end_date, api, limit=limit)
        df["NASDAQ"] = df2.c 
    
    if "VIX" in feature_columns:
        df2 = get_alpaca_data("VIXY", end_date, api, limit=limit)
        df["VIX"] = df2.c

    if "7MA" in feature_columns:
        df["7MA"] = df.c.rolling(window=7).mean()
    
    if ("upper_band" or "lower_band") in feature_columns:
        upperband, middleband, lowerband = ta.BBANDS(df.c, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
        df["upper_band"] = upperband
        df["lower_band"] = lowerband

    if "OBV" in feature_columns:
        df["OBV"] = ta.OBV(df.c, df.v)

    if "RSI" in feature_columns:
        df["RSI"] = ta.RSI(df.c)
    
    if "lin_reg" in feature_columns:
        df["lin_reg"] = ta.LINEARREG(df.c, timeperiod=14)

    if "lin_reg_angle" in feature_columns:
        df["lin_reg_angle"] = ta.LINEARREG_ANGLE(df.c, timeperiod=14)

    if "lin_reg_intercept" in feature_columns:
        df["lin_reg_intercept"] = ta.LINEARREG_INTERCEPT(df.c, timeperiod=14)

    if "lin_reg_slope" in feature_columns:
        df["lin_reg_slope"] = ta.LINEARREG_SLOPE(df.c, timeperiod=14)

    if "pearsons_correl" in feature_columns:
        df["pearsons_correl"] = ta.CORREL(df.h, df.l, timeperiod=30)

    if "money_flow_ind" in feature_columns:
        df["money_flow_ind"] = ta.MFI(df.h, df.l, df.c, df.v, timeperiod=14)

    if "wills_r" in feature_columns:
        df["wills_r"] = ta.WILLR(df.h, df.l, df.c, timeperiod=14)

    if "std_dev" in feature_columns:
        df["std_dev"] = ta.STDDEV(df.c, timeperiod=5, nbdev=1)

    if ("min" or "max") in feature_columns:
        minimum, maximum = ta.MINMAX(df.c, timeperiod=30)
        df["min"] = minimum
        df["max"] = maximum

    if "commodity_channel_ind" in feature_columns:
        df["commodity_channel_ind"] = ta.CCI(df.h, df.l, df.c, timeperiod=14)

    if "parabolic_SAR" in feature_columns:
        df["parabolic_SAR"] = ta.SAR(df.h, df.l)

    if "parabolic_SAR_extended" in feature_columns:
        df["parabolic_SAR_extended"] = ta.SAREXT(df.h, df.l, startvalue=0, offsetonreverse=0, accelerationinitlong=0, 
        accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    if "rate_of_change" in feature_columns:
        df["rate_of_change"] = ta.ROC(df.c, timeperiod=10)

    if "ht_dcperiod" in feature_columns:
        df["ht_dcperiod"] = ta.HT_DCPERIOD(df.c)

    if "ht_trendmode" in feature_columns:
        df["ht_trendmode"] = ta.HT_TRENDMODE(df.c)

    if "ht_dcphase" in feature_columns:
        df["ht_dcphase"] = ta.HT_DCPHASE(df.c)

    if ("ht_inphase" or "quadrature") in feature_columns:
        df["ht_inphase"], df["quadrature"] = ta.HT_PHASOR(df.c)

    if ("ht_sine" or "ht_leadsine") in feature_columns:
        df["ht_sine"], df["ht_leadsine"] = ta.HT_SINE(df.c)

    if "ht_trendline" in feature_columns:
        df["ht_trendline"] = ta.HT_TRENDLINE(df.c)

    if "momentum" in feature_columns:
        df["momentum"] = ta.MOM(df.c, timeperiod=10)

    if "abs_price_osc" in feature_columns:
        df["abs_price_osc"] = ta.APO(df.c, fastperiod=12, slowperiod=26, matype=0)

    if "KAMA" in feature_columns:
        df["KAMA"] = ta.KAMA(df.c, timeperiod=30)

    if "typical_price" in feature_columns:    
        df["typical_price"] = ta.TYPPRICE(df.h, df.l, df.c)

    if "ultimate_osc" in feature_columns:
        df["ultimate_osc"] = ta.ULTOSC(df.h, df.l, df.c, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    if "chaikin_line" in feature_columns:
        df["chaikin_line"] = ta.AD(df.h, df.l, df.c, df.v)

    if "chaikin_osc" in feature_columns:
        df["chaikin_osc"] = ta.ADOSC(df.h, df.l, df.c, df.v, fastperiod=3, slowperiod=10)

    if "norm_average_true_range" in feature_columns:
        df["norm_average_true_range"] = ta.NATR(df.h, df.l, df.c, timeperiod=14)

    if "median_price" in feature_columns:
        df["median_price"] = ta.MEDPRICE(df.h, df.l)

    if "variance" in feature_columns:
        df["variance"] = ta.VAR(df.c, timeperiod=5, nbdev=1)

    if ("aroon_down" or "aroon_up") in feature_columns:
        df["aroon_down"], df["aroon_up"] = ta.AROON(df.h, df.l, timeperiod=14)

    if "aroon_osc" in feature_columns:
        df["aroon_osc"] = ta.AROONOSC(df.h, df.l, timeperiod=14)

    if "balance_of_pow" in feature_columns:
        df["balance_of_pow"] = ta.BOP(df.o, df.h, df.l, df.c)

    if "chande_momen_osc" in feature_columns:    
        df["chande_momen_osc"] = ta.CMO(df.c, timeperiod=14)

    if ("macd" or "macdsignal" or "macdhist") in feature_columns:
        df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(df.c, fastperiod=12, slowperiod=26, signalperiod=9)

    if ("control_MACD" or "control_MACD_signal" or "control_MACD_hist") in feature_columns:
        df["control_MACD"], df["control_MACD_signal"], df["control_MACD_hist"] = ta.MACDEXT(df.c, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

    if ("fix_MACD" or "fix_MACD_signal" or "fix_MACD_hist") in feature_columns:
        df["fix_MACD"], df["fix_MACD_signal"], df["fix_MACD_hist"] = ta.MACDFIX(df.c, signalperiod=9)

    if "minus_directional_ind" in feature_columns:
        df["minus_directional_ind"] = ta.MINUS_DI(df.h, df.l, df.c, timeperiod=14)

    if "minus_directional_move" in feature_columns:
        df["minus_directional_move"] = ta.MINUS_DM(df.h, df.l, timeperiod=14)

    if "plus_directional_ind" in feature_columns:
        df["plus_directional_ind"] = ta.PLUS_DI(df.h, df.l, df.c, timeperiod=14)

    if "plus_directional_move" in feature_columns:
        df["plus_directional_move"] = ta.PLUS_DM(df.h, df.l, timeperiod=14)

    if "percentage_price_osc" in feature_columns:
        df["percentage_price_osc"] = ta.PPO(df.c, fastperiod=12, slowperiod=26, matype=0)

    if ("stochas_fast_k" or "stochas_fast_d") in feature_columns:
        df["stochas_fast_k"], df["stochas_fast_d"] = ta.STOCHF(df.h, df.l, df.c, fastk_period=5, fastd_period=3, fastd_matype=0)

    if ("stochas_relative_strength_k" or "stochas_relative_strength_d") in feature_columns:
        df["stochas_relative_strength_k"], df["stochas_relative_strength_d"] = ta.STOCHRSI(df.c, fastk_period=5, fastd_period=3, fastd_matype=0)

    if ("stochas_slowk" or "stochas_slowd") in feature_columns:
        df["stochas_slowk"], df["stochas_slowd"] = ta.STOCH(df.h, df.l, df.c, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    if "TRIX" in feature_columns:
        df["TRIX"] = ta.TRIX(df.c, timeperiod=30)

    if "weighted_moving_avg" in feature_columns:
        df["weighted_moving_avg"] = ta.WMA(df.c, timeperiod=30)

    if ("upband" or "midband" or "lowband") in feature_columns:
        df["upband"], df["midband"], df["lowband"] = ta.BBANDS(df.c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

    if "double_exponetial_moving_avg" in feature_columns:
        df["double_exponetial_moving_avg"] = ta.DEMA(df.c, timeperiod=30)

    if "EMA" in feature_columns:
        df["EMA"] = ta.EMA(df.c, timeperiod=5)

    if ("MESA_mama" or "MESA_fama") in feature_columns:
        df["MESA_mama"], df["MESA_fama"] = ta.MAMA(df.c)

    if "midpoint" in feature_columns:
        df["midpoint"] = ta.MIDPOINT(df.c, timeperiod=14)

    if "midprice" in feature_columns:
        df["midprice"] = ta.MIDPRICE(df.h, df.l, timeperiod=14)

    if "triple_exponential_moving_avg" in feature_columns:
        df["triple_exponential_moving_avg"] = ta.TEMA(df.c, timeperiod=30)

    if "triangular_moving_avg" in feature_columns:
        df["triangular_moving_avg"] = ta.TRIMA(df.c, timeperiod=30)

    if "avg_directional_movement_index" in feature_columns:
        df["avg_directional_movement_index"] = ta.ADX(df.h, df.l, df.c, timeperiod=14)

    if "true_range" in feature_columns:
        df["true_range"] = ta.TRANGE(df.h, df.l, df.c)

    if "avg_price" in feature_columns:
        df["avg_price"] = ta.AVGPRICE(df.o, df.h, df.l, df.c)

    if "weighted_c_price" in feature_columns:
        df["weighted_c_price"] = ta.WCLPRICE(df.h, df.l, df.c)

    if "beta" in feature_columns:
        df["beta"] = ta.BETA(df.h, df.l, timeperiod=5)

    if "TSF" in feature_columns:
        df["TSF"] = ta.TSF(df.c, timeperiod=17)

    if "day_of_week" in feature_columns:
        df = convert_date_values(df)

    if "v" not in feature_columns:
        df = df.drop(columns=["v"])

    if "o" not in feature_columns:
        df = df.drop(columns=["o"])

    if "l" not in feature_columns:
        df = df.drop(columns=["l"])

    if "h" not in feature_columns:
        df = df.drop(columns=["h"])

    if "m" not in feature_columns:
        df = df.drop(columns=["m"])

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

def aquire_preprocess_data(symbol, params, end_date=None, scale=True, to_print=True):
    if to_print:
        print(f"""Included features: {params["FEATURE_COLUMNS"]}""")

    if end_date is not None:
        df = make_dataframe(symbol, params["FEATURE_COLUMNS"], params["LIMIT"], end_date, to_print)
    else:
        df = make_dataframe(symbol, params["FEATURE_COLUMNS"], params["LIMIT"], to_print=to_print)
    
    for col in params["FEATURE_COLUMNS"]:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    result = {}
    result["df"] = df.copy()

    # print("future: " + str(df["future"]))
    # print("future: " + str(df["future"][0]))
    # print("future: " + str(df["future"].values))
    # if params["LOSS"] == "huber_loss":
    #             print(df["future"].values)
    #             df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1), df["future"])

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in df.columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        result["column_scaler"] = column_scaler

    if params["LOSS"] == "huber_loss":
        df["future"] = df[params["TEST_VAR"]].shift(-params["LOOKUP_STEP"])
    else:
        df["future"] = df[params["TEST_VAR"]].shift(-params["LOOKUP_STEP"])
        df["future"] = list(map(lambda current, future: int(float(future) > float(current)), df.c, df.future))
    # add the target column (label) by shifting by `lookup_step`\

    if "c" not in params["FEATURE_COLUMNS"]:
        df = df.drop(columns=["c"])

    return df, result

def load_data(symbol, params, end_date=None, shuffle=True, scale=True, to_print=True):
    df, result = aquire_preprocess_data(symbol, params, end_date, scale, to_print)
    
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
    # print(last_sequence)
    # construct the X"s and y"s
    X, y = [], []
    
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    
    # split the dataset
    result["X_train"], result["X_valid"], result["y_train"], result["y_valid"] = train_test_split(X, y, test_size=params["TEST_SIZE"], shuffle=shuffle)
    result["X_valid"], result["X_test"], result["y_valid"], result["y_test"] = train_test_split(result["X_valid"], result["y_valid"], test_size=.006, shuffle=shuffle)

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

def load_2D_data(symbol, params, end_date=None, shuffle=True, scale=True, to_print=True):
    df, result = aquire_preprocess_data(symbol, params, end_date, scale, to_print)

    df.dropna(inplace=True)
    y = df["future"]
    df = df.drop(columns="future")
    X = df.to_numpy()

    X = np.array(X)
    y = np.array(y)
    
    result["X_train"], result["X_valid"], result["y_train"], result["y_valid"] = train_test_split(X, y, test_size=params["TEST_SIZE"], shuffle=shuffle)
    result["X_valid"], result["X_test"], result["y_valid"], result["y_test"] = train_test_split(result["X_valid"], result["y_valid"], test_size=.006, shuffle=shuffle)

    return result
