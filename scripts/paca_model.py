from config.silen_ten import silence_tensorflow
silence_tensorflow()
from functions.functions import delete_files_in_folder, check_model_folders, get_model_name
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from config.environ import (directory_dict, random_seed, save_logs,
defaults, load_params)
from tensorflow.keras.layers import LSTM
from functions.time_functs import get_time_string, get_past_date_string
from functions.functions import get_model_name, layer_name_converter
from functions.paca_model_functs import create_model, get_accuracy, get_all_accuracies, get_current_price, load_model_with_data, predict, return_real_predict
from functions.data_load_functs import load_3D_data, make_dataframe, load_2D_data
from functions.io_functs import save_prediction, load_saved_predictions
from scipy.signal import savgol_filter
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statistics import mean
import pandas as pd
import talib as ta
import numpy as np
import socket
import random
import os


def nn_train_save(symbol, params=defaults, end_date=None, predictor="nn1"):
    #description of all the parameters used is located inside environment.py
    tf.keras.backend.clear_session()
    tf.keras.backend.reset_uids()

    options = {"shape_optimization": True}
    tf.config.optimizer.set_experimental_options(options)
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2, --tf_xla_cpu_global_jit" # turns on xla and cpu xla
    tf.config.optimizer.set_jit(True)

    # set seed, so we can get the same results after rerunning several times
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)

    nn_params = params[predictor]
    
    check_model_folders(params["SAVE_FOLDER"], symbol)
    
    # model name to save, making it as unique as possible based on parameters
    model_name = (symbol + "-" + get_model_name(nn_params))

    first_layer = layer_name_converter(nn_params["LAYERS"][0])
    if first_layer == "Dense":
        data, train, valid, test = load_2D_data(symbol, nn_params, end_date, tensorify=True)
    else:
        data, train, valid, test = load_3D_data(symbol, nn_params, end_date)

    model = create_model(nn_params)

    logs_dir = "logs/" + get_time_string() + "-" + params["SAVE_FOLDER"]

    checkpointer = ModelCheckpoint(directory_dict["model"] + "/" + params["SAVE_FOLDER"] + "/" 
        + model_name + ".h5", save_weights_only=True, save_best_only=True, verbose=1)
    
    if save_logs:
        tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch="200, 1200") 
    else:
        tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch=0)

    early_stop = EarlyStopping(patience=nn_params["PATIENCE"])
    
    history = model.fit(train,
        batch_size=nn_params["BATCH_SIZE"],
        epochs=nn_params["EPOCHS"],
        verbose=2,
        validation_data=valid,
        callbacks = [tboard_callback, checkpointer, early_stop]   
    )

    epochs_used = len(history.history["loss"])
        
    if not save_logs:
        delete_files_in_folder(logs_dir)
        os.rmdir(logs_dir)

    return epochs_used

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def nn_load_predict(symbol, params, current_date, predictor, to_print=False):
    data, model = load_model_with_data(symbol, current_date, params, predictor, to_print)
    predicted_price = predict(model, data, params[predictor]["N_STEPS"], params[predictor]["TEST_VAR"],
        layer=params[predictor]["LAYERS"][0])

    return predicted_price

def ensemble_predictor(symbol, params, current_date):
    pd.set_option("display.max_columns", None)
    ensemb_predict_list = []

    epochs_dict = {}
    df = make_dataframe(symbol, load_params["FEATURE_COLUMNS"], limit=load_params["LIMIT"], 
        end_date=current_date, to_print=False)

    if not params["TRADING"]:
        if current_date:
            s_current_date = get_past_date_string(current_date)
        for predictor in params["ENSEMBLE"]:
            if "nn" in predictor:
                epochs_dict[predictor] = 0
                if not symbol in params[predictor]["SAVE_PRED"]:
                    params[predictor]["SAVE_PRED"][symbol] = {}
                if not s_current_date in params[predictor]["SAVE_PRED"][symbol]:
                    params[predictor]["SAVE_PRED"][symbol][s_current_date] = {}

    for predictor in params["ENSEMBLE"]:
        if predictor == "7MA":
            df["7MA"] = df.c.rolling(window=7).mean()
            predicted_price = np.float32(df["7MA"][len(df.c) - 1])
            ensemb_predict_list.append(predicted_price)
            
        elif predictor == "lin_reg":
            df["lin_reg"] = ta.LINEARREG(df.c, timeperiod=7)
            predicted_price = np.float32(df.lin_reg[len(df.c) - 1])
            ensemb_predict_list.append(predicted_price)

        elif predictor == "sav_gol":
            df["sc"] = savgol_filter(df.c, 7, 3)
            predicted_price = np.float32(df.sc[len(df.c) - 1])
            ensemb_predict_list.append(predicted_price)

        elif predictor == "EMA":
            df["EMA"] = ta.EMA(df.c, timeperiod=5)
            predicted_price = np.float32(df["EMA"][len(df.c) - 1])
            ensemb_predict_list.append(predicted_price)

        elif "DTREE" in predictor:
            df2D = load_2D_data(symbol, params[predictor], current_date, shuffle=True, 
                scale=True, to_print=False)
            tree = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
                min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"])
            tree.fit(df2D["X_train"], df2D["y_train"])
            df2D = load_2D_data(symbol, params[predictor], current_date, shuffle=False, 
                scale=True, to_print=False)
            tree_pred = tree.predict(df2D["X_test"])
            scale = df2D["column_scaler"][params[predictor]["TEST_VAR"]]
            tree_pred = np.array(tree_pred)
            tree_pred = tree_pred.reshape(1, -1)
            predicted_price = np.float32(scale.inverse_transform(tree_pred)[-1][-1])
            ensemb_predict_list.append(predicted_price)

        elif "RFORE" in predictor:
            df2D = load_2D_data(symbol, params[predictor], current_date, shuffle=True, 
                scale=True, to_print=False)
            fore = RandomForestRegressor(n_estimators=100)
            fore.fit(df2D["X_train"], df2D["y_train"])
            df2D = load_2D_data(symbol, params[predictor], current_date, shuffle=False, 
                scale=True, to_print=False)
            fore_pred = fore.predict(df2D["X_test"])
            scale = df2D["column_scaler"][params[predictor]["TEST_VAR"]]
            fore_pred = np.array(fore_pred)
            fore_pred = fore_pred.reshape(1, -1)
            predicted_price = np.float32(scale.inverse_transform(fore_pred)[-1][-1])
            ensemb_predict_list.append(predicted_price)

        elif "nn" in predictor:
            if params["TRADING"]:
                predicted_price = nn_load_predict(symbol, params, current_date,  predictor)
            else:
                if load_saved_predictions(symbol, params, current_date, predictor):
                    predicted_price, epochs_run = load_saved_predictions(symbol, params, current_date, predictor)
                    epochs_dict[predictor] = epochs_run
                else:
                    epochs_run = nn_train_save(symbol, params, current_date, predictor)
                    epochs_dict[predictor] = epochs_run
                    predicted_price = nn_load_predict(symbol, params, current_date, predictor)
                    save_prediction(symbol, params, current_date, predictor, predicted_price, epochs_run)
            ensemb_predict_list.append(np.float32(predicted_price))

    print(ensemb_predict_list)
    # print(f"ensemb predict list {ensemb_predict_list}")
    final_prediction = mean(ensemb_predict_list)
    print(f"final pred {final_prediction}")
    current_price = get_current_price(df)

    return final_prediction, current_price, epochs_dict


def ensemble_accuracy(symbol, params, current_date, classification=False):
    for predictor in params:
        if "nn" in predictor:
            pass
            data, model = load_model_with_data(symbol, current_date, params, predictor)

            # get_all_accuracies(model, data, params[predictor["LOOKUP_STEP"]], params[predictor["TEST_VAR"]])

            y_train_real, y_train_pred = return_real_predict(model, data["X_train"], data["y_train"], 
            data["column_scaler"][params[predictor["TEST_VAR"]]], classification)
            print(f"{y_train_real, y_train_pred}")
            y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], 
            data["column_scaler"][params[predictor["TEST_VAR"]]], classification)
            y_test_real, y_test_pred = return_real_predict(model, data["X_test"], data["y_test"],
            data["column_scaler"][params[predictor["TEST_VAR"]]], classification)





    train_acc = get_accuracy(y_train_real, y_train_pred, params[predictor["LOOKUP_STEP"]])
    valid_acc = get_accuracy(y_valid_real, y_valid_pred, params[predictor["LOOKUP_STEP"]])
    test_acc = get_accuracy(y_test_real, y_test_pred, params[predictor["LOOKUP_STEP"]])
    




    return train_acc, valid_acc, test_acc 
