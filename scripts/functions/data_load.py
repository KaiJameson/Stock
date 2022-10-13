from matplotlib import ticker
from config.environ import directory_dict
from config.api_key import alpha_key
from functions.time import get_past_date_string
from functions.error import  net_error_handler, keyboard_interrupt
from functions.trade import get_api
from functions.time import modify_timestamp, get_current_datetime, get_past_datetime
from functions.functions import layer_name_converter
from functions.technical_indicators import techs_dict
from functions.io import save_to_dictionary, read_saved_contents
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental import AUTOTUNE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter, cwt
from collections import deque
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import numpy as np
import time
import copy
import requests
import os
import sys




def modify_dataframe(symbol, features, df, test_var, option, to_print):
    base_features = ["o", "c", "l", "h", "v", "tc", "vwap"]

    for feature in features:
        if feature not in base_features:
            if "." in feature:
                dot_split = feature.split(".")
                if dot_split[0] in techs_dict:
                    techs_dict[dot_split[0]]["function"](dot_split[0], dot_split[1], df, symbol)
                elif dot_split[0].startswith("tick-"):
                    feature_split = dot_split[0].split("-")
                    print(f"feature split {feature_split}")
                    ticker_df = get_proper_df(feature_split[1], 4000, option)
                    # ticker_df = df_subset(current_date, ticker_df)

                    if feature_split[2] not in base_features:
                        if feature_split[2] in techs_dict:
                            techs_dict[feature_split[2]]["function"](feature_split[2], dot_split[1], ticker_df, feature_split[1])
                            df[feature] = ticker_df[f"{feature_split[2]}.{dot_split[1]}"]
                    else:
                        df[feature] = ticker_df[feature_split[2]]
            else:
                if feature in techs_dict:
                    techs_dict[feature]["function"](feature, df, symbol)
                elif feature.startswith("tick-"):
                    feature_split = feature.split("-")
                    ticker_df = get_proper_df(feature_split[1], 4000, option)
                    # ticker_df = df_subset(current_date, ticker_df)

                    if feature_split[2] not in base_features:
                        if feature_split[2] in techs_dict:
                            print(f"did we get herer \n\n\n")
                            techs_dict[feature_split[2]]["function"](feature_split[2], ticker_df, feature_split[1])
                            df[feature] = ticker_df[feature_split[2]]
                    else:
                        df[feature] = ticker_df[feature_split[2]]
            
            if feature not in df.columns:
                print(f"Feature {feature} is not in the technical indicators dictionary. That sucks, probably")
       
    
    for feature in list(df.columns):
        if feature not in features and feature != test_var and feature != "c":
            # print(f"dropping {feature}")
            df = df.drop(columns=[feature])

    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    pd.set_option('display.expand_frame_repr', False)
    if to_print:
        print(df.head(2))
        print(df.tail(2))
        # print(df.shape)

    return df

def scale_data(df, result):
    column_scaler = {}
    for column in df.columns:
        scaler = preprocessing.MinMaxScaler()
        df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler

    result["column_scaler"] = column_scaler
    return result

def get_alpacaV2_df(symbol, limit=1000, to_print=True):
    api = get_api()
    no_connection = True
    end = get_current_datetime()
    start = get_past_datetime(2000, 1, 1)
    while no_connection:
        try:
            s = time.perf_counter()
            df = api.get_bars(symbol, start=start, end=end, timeframe=TimeFrame.Day, 
                limit=limit).df
            df = df.rename(columns={"open": "o", "high":"h", "low": "l", "close": "c",
                "volume": "v", "trade_count": "tc"})
            df.index = df.index.date
            no_connection = False
            # print(f"all loading took {time.perf_counter() - s}")
        except KeyboardInterrupt:
            keyboard_interrupt()
        except Exception:
            net_error_handler(symbol, Exception)

    return df

def get_alpha_df(symbol, output_size):
    load_dictionary = {
        "o": {},
        "h": {},
        "l": {},
        "c": {},
        "v": {},
    }
    # print(f"""{directory_dict["data"]}/{symbol}.txt""")
    if os.path.isfile(f"""{directory_dict["data"]}/{symbol}.txt"""):
        load_dictionary = read_saved_contents(f"""{directory_dict["data"]}/{symbol}.txt""", load_dictionary)
        df = pd.DataFrame(load_dictionary, dtype=np.float32)
        # TODO check for whether or not this is up to date and then update it
    else:
        connection = False
        # print("hehe")
        while not connection:
            # print("hello", flush=True)
            try:
                df = download_alpha_df(symbol, output_size)
                connection = True
            except KeyboardInterrupt:
                keyboard_interrupt()
            except Exception:
                net_error_handler(symbol, Exception)
    
    df = df.iloc[::-1]
    print(df.head(1))
    print(df.tail(1))
    return df

def download_alpha_df(symbol, output_size):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={alpha_key}"
    no_connection = True
    while no_connection:
        try:
            r = requests.get(url)
            no_connection = False
        except KeyboardInterrupt:
            keyboard_interrupt()
        except Exception:
            net_error_handler(symbol, Exception)

    data = r.json()
    # print(data)
    df = pd.DataFrame(data["Time Series (Daily)"], dtype=np.float32)
    df = df.transpose()
    df = df.rename(columns={"1. open": "o", "2. high": "h", "3. low": "l", "4. close": "c", "5. volume": "v"})
    # print(df.head(1))
    # print(df.tail(1))
    df_dict = df.to_dict()
    save_to_dictionary(f"""{directory_dict["data"]}/{symbol}.txt""", df_dict)
    
    return df

def get_proper_df(symbol, limit, option):
    if option == "alp":
        df = get_alpha_df(symbol, "full")
    elif option == "V2":
        df = get_alpacaV2_df(symbol, limit, to_print=True)

    return df

def get_df_dict(symbol, params, option, to_print):
    df_dict = {}

    base_df = get_proper_df(symbol, 4000, option)
    df_dict['price'] = modify_dataframe(symbol, "c", base_df, "c", option, False)

    for predictor in params['ENSEMBLE']:
        df_dict[predictor] = modify_dataframe(symbol, params[predictor]["FEATURE_COLUMNS"], base_df, params[predictor]["TEST_VAR"], option, to_print)

    return df_dict
        

def load_all_data(params, df_dict, split_type, to_print=True):
    data_dict = {}
    req_2d = ["DTREE", "XTREE", "BAGREG", "RFORE", "KNN", "ADA", "XGB", "MLENS", "MLP"]
    
    for predictor in params['ENSEMBLE']:
        in_req_2d = [bool(i) for i in req_2d if i in predictor]
        if len(in_req_2d) > 0:
            result = load_2D_data(params[predictor], df_dict[predictor], split_type, shuffle=True, tensorify=False, to_print=to_print)
            data_dict[predictor] = result
        if "nn" in predictor:
            if layer_name_converter(params[predictor]['LAYERS'][0]) == "Dense":
                result, train, valid, test = load_2D_data(params[predictor], df_dict[predictor], split_type, params[predictor]["SHUFFLE"],
                    tensorify=True, to_print=to_print)
                data_dict[predictor] = {"result": result, "train": train, "valid": valid,
                    "test": test}
            else:
                result, train, valid, test = load_3D_data(params[predictor], df_dict[predictor], split_type, params[predictor]["SHUFFLE"],
                to_print=to_print)
                data_dict[predictor] = {"result": result, "train": train, "valid": valid,
                    "test": test}
                

    return data_dict

def preprocess_dfresult(params, df, scale, to_print):
    tt_df = copy.deepcopy(df)
    tt_df = tt_df.replace(0.000000, 0.000000001)
    if to_print:
        print(f"""Included features: {params["FEATURE_COLUMNS"]}""")
   
    # tt_df = modify_dataframe(symbol, params["FEATURE_COLUMNS"], tt_df, current_date, params["TEST_VAR"], to_print)
    for col in params["FEATURE_COLUMNS"]:
        assert col in tt_df.columns, f"'{col}' does not exist in the dataframe."

    if params['TEST_VAR'] == "acc":
        future = list(map(lambda current, future: int(float(future) > float(current)), tt_df['c'][:-params["LOOKUP_STEP"]], tt_df['c'][params["LOOKUP_STEP"]:]))
        future.insert(len(future), np.nan)
        tt_df["future"] = future
    else:
        tt_df["future"] = tt_df[params["TEST_VAR"]].shift(-params["LOOKUP_STEP"])

    # print(tt_df)

    if params['TEST_VAR'] not in params['FEATURE_COLUMNS'] and params['TEST_VAR'] in tt_df.columns:
        tt_df = tt_df.drop(columns=[params['TEST_VAR']])

    result = {}
    if scale:
        result = scale_data(tt_df, result)

    return tt_df, result

def construct_3D_np(tt_df, params, result):
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(tt_df[params["FEATURE_COLUMNS"]].tail(params["LOOKUP_STEP"]))
    # drop NaNs
    tt_df = tt_df.dropna()
    sequence_data = []
    sequences = deque(maxlen=params["N_STEPS"])
    for entry, target in zip(tt_df[params["FEATURE_COLUMNS"]].values, tt_df["future"].values):
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

    return X, y

def load_3D_data(params, df, split_type, shuffle=True, scale=True, to_print=True):
    tt_df, result = preprocess_dfresult(params, df, scale=scale, to_print=to_print)
    
    X, y = construct_3D_np(tt_df, params, result)

    print(f"Final X data shape is {X.shape}")
    result = split_data(X, y, params["TEST_SIZE"], split_type, shuffle, result)

    train, valid, test = make_tensor_slices(params, result)
    return result, train, valid, test
   


def load_2D_data(params, df, split_type, shuffle=True, scale=True, tensorify=False, to_print=True):
    tt_df, result = preprocess_dfresult(params, df, scale, to_print)

    tt_df = tt_df.dropna()
    y = tt_df["future"]
    tt_df = tt_df.drop(columns="future")
    X = tt_df.to_numpy()

    X = np.array(X)
    y = np.array(y)

    print(f"Final X data shape is {X.shape}")
    
    result = split_data(X, y, params["TEST_SIZE"], split_type, shuffle, result)   

    if tensorify:
        train, valid, test = make_tensor_slices(params, result)
        return result, train, valid, test
    else:
        return result

def split_data(X, y, test_size, split_type, shuffle, result):
    if split_type == "one_day":
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=2, random_state=42, 
            shuffle=False)
        result["X_train"], result["X_valid"], result["y_train"], result["y_valid"] = train_test_split(result["X_train"], result["y_train"],
            test_size=test_size, random_state=42, shuffle=shuffle)
    elif type(split_type) == type(1): # In this case split_type should be an int
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=split_type, random_state=42, 
            shuffle=False)
        result["X_train"], result["X_valid"], result["y_train"], result["y_valid"] = train_test_split(result["X_train"], result["y_train"],
            test_size=test_size, random_state=42, shuffle=shuffle)
    else:
        print(f"split_type {split_type} is improperly defined")
        print(f"Exiting now")
        sys.exit(-1)
    return result

def make_tensor_slices(params, result):
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

    return train, valid, test

def df_subset(df_dict, current_date):
    df_sub_dict = {}
    for df in df_dict:
        df_sub = copy.deepcopy(df_dict[df])
        df_sub = df_sub[df_sub.index <= current_date]

        df_sub_dict[df] = df_sub
    
    return df_sub_dict


