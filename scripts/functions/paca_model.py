from tensorflow.python.keras.layers.core import Activation
from config.silen_ten import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from sklearn.metrics import accuracy_score
from config.api_key import intrinio_sandbox_key
from functions.functions import layer_name_converter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from intrinio_sdk.rest import ApiException
import alpaca_trade_api as tradeapi
import talib as ta
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import intrinio_sdk as intrinio
import time
import sys
import copy



def create_model(params):
    model = Sequential()

    for layer in range(len(params["LAYERS"])):
        if layer == 0:
            model_first_layer(model, params["LAYERS"], layer, params["N_STEPS"], params["FEATURE_COLUMNS"], params["BIDIRECTIONAL"])
        elif layer == len(params["LAYERS"]) - 1:
            model_last_layer(model, params["LAYERS"], layer, params["BIDIRECTIONAL"])
        else:
            model_hidden_layers(model, params["LAYERS"], layer, params["BIDIRECTIONAL"])
    
        model.add(Dropout(params["DROPOUT"]))

    if params['TEST_VAR'] == "acc":
        model.add(Dense(1))
        if params['LOSS'] == "binary_crossentropy":
            model.compile(loss=BinaryCrossentropy(from_logits=True), metrics=["accuracy"], optimizer=params["OPTIMIZER"])
        else:
            model.compile(loss=BinaryFocalCrossentropy(from_logits=True), metrics=["accuracy"], optimizer=params["OPTIMIZER"])
    else:
        model.add(Dense(1, activation="linear"))
        model.compile(loss=params["LOSS"], metrics=["mean_absolute_error"], optimizer=params["OPTIMIZER"])

    return model

def model_first_layer(model, layers, ind, n_steps, features, bidirectional):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if layer_name == "Dense":
        model.add(layers[ind][1](layers[ind][0], activation="elu", input_shape=(None, len(features))))
    else:
        if next_layer_name == "Dense":
            if bidirectional:
                model.add(Bidirectional(layers[ind][1](layers[ind][0], return_sequences=False), 
                    input_shape=(n_steps, len(features))))
            else:
                model.add(layers[ind][1](layers[ind][0], return_sequences=False, 
                    input_shape=(n_steps, len(features))))
        else:
            if bidirectional:
                model.add(Bidirectional(layers[ind][1](layers[ind][0], return_sequences=True), 
                    input_shape=(n_steps, len(features))))
            else:
                model.add(layers[ind][1](layers[ind][0], return_sequences=True, 
                    input_shape=(n_steps, len(features))))

    return model

def model_hidden_layers(model, layers, ind, bidirectional):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if layer_name == "Dense":
        model.add(layers[ind][1](layers[ind][0], activation="elu"))
    else:
        if next_layer_name == "Dense":
            if bidirectional:
                model.add(Bidirectional(layers[ind][1](layers[ind][0], return_sequences=False)))
            else:
                model.add(layers[ind][1](layers[ind][0], return_sequences=False))
        else:
            if bidirectional:
                model.add(Bidirectional(layers[ind][1](layers[ind][0], return_sequences=True)))
            else:
                model.add(layers[ind][1](layers[ind][0], return_sequences=True))

    return model

def model_last_layer(model, layers, ind, bidirectional):
    layer_name = layer_name_converter(layers[ind])

    if layer_name == "Dense":
        model.add(layers[ind][1](layers[ind][0], activation="elu"))
    else:
        if bidirectional:
            model.add(Bidirectional(layers[ind][1](layers[ind][0], return_sequences=False)))
        else:
            model.add(layers[ind][1](layers[ind][0], return_sequences=False))
    
    return model


def predict(model, data, n_steps, test_var="c", layer="LSTM"):
    if test_var == "acc":
        last_sequence = data["last_sequence"][:n_steps]
        last_sequence = np.expand_dims(last_sequence, axis=0)

        predicted_val = model.predict(last_sequence)[-1][-1]
        
        if predicted_val > 0:
            predicted_val = 1
        else:
            predicted_val = 0

    else:
        column_scaler = data["column_scaler"]
        
        if layer_name_converter(layer) != "Dense":
            last_sequence = data["last_sequence"][:n_steps]
            # reshape the last sequence
            # last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
            # expand dimension
            last_sequence = np.expand_dims(last_sequence, axis=0)
            # get the prediction (scaled from 0 to 1)
            prediction = model.predict(last_sequence)
            predicted_val = column_scaler["future"].inverse_transform(prediction)[0][0]
            
        else: 
            prediction = model.predict(data["X_test"])
            pred = np.array(prediction)
            pred= pred.reshape(1, -1)
            predicted_val = column_scaler["future"].inverse_transform(pred)[-1][-1]
    return predicted_val


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
    y = data["c"]
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
    
def get_all_accuracies(model, data, lookup_step, test_var="c", classification=False):
    y_train_real, y_train_pred = return_real_predict(model, data["X_train"], data["y_train"], 
    data["column_scaler"][test_var], classification)
    train_acc = get_accuracy(y_train_real, y_train_pred, lookup_step)
    y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], 
    data["column_scaler"][test_var], classification)
    valid_acc = get_accuracy(y_valid_real, y_valid_pred, lookup_step)
    y_test_real, y_test_pred = return_real_predict(model, data["X_test"], data["y_test"],
        data["column_scaler"][test_var], classification)
    test_acc = get_accuracy(y_test_real, y_test_pred, lookup_step)

    return train_acc, valid_acc, test_acc 

def get_accuracy(y_real, y_pred, lookup_step):
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_real[:-lookup_step], y_pred[lookup_step:]))
    y_real = list(map(lambda current, future: int(float(future) > float(current)), y_real[:-lookup_step], y_real[lookup_step:]))

    return accuracy_score(y_real, y_pred)

def get_percent_away(y_real, y_pred):
    the_diffs = []
    for i in range(len(y_real) - 1):
        per_diff = (abs(y_real[i] - y_pred[i])/y_real[i]) * 100
        the_diffs.append(per_diff)
    pddf = pd.DataFrame(data=the_diffs)
    pddf = pddf.values
    return round(pddf.mean(), 2)

def get_all_maes(model, test_tensorslice, valid_tensorslice, train_tensorslice, data):
    train_mae = get_mae(model, train_tensorslice, data)
    valid_mae = get_mae(model, valid_tensorslice, data)
    test_mae =  get_mae(model, test_tensorslice, data)

    return test_mae, valid_mae, train_mae

def get_mae(model, tensorslice, data, test_var="close"):
    mse, mae = model.evaluate(tensorslice, verbose=0)
    mae = data["column_scaler"][test_var].inverse_transform([[mae]])[0][0]

    return mae

def return_real_predict(model, X_data, y_data, column_scaler, classification=False):
    y_pred = model.predict(X_data)
    y_real = np.squeeze(column_scaler.inverse_transform(np.expand_dims(y_data, axis=0)))
    if not classification:
        y_pred = np.squeeze(column_scaler.inverse_transform(y_pred))

    return y_real, y_pred

def get_current_price(df):
    return df.c[-1]

