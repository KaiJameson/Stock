from config.silen_ten import silence_tensorflow
silence_tensorflow()
from functions.functions import delete_files_in_folder, check_model_folders, get_model_name
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from config.environ import (directory_dict, random_seed, save_logs,
defaults, load_params)
from tensorflow.keras.layers import LSTM
from functions.time_functs import get_time_string, get_past_datetime
from functions.functions import get_model_name
from functions.paca_model_functs import create_model, nn_report, get_all_accuracies, get_all_maes, get_current_price, load_model_with_data, predict
from functions.data_load_functs import load_data
from statistics import mean
import talib as ta
import numpy as np
import socket
import random
import os


def nn_train_save(symbol, end_date=None, params=defaults, save_folder="trading"):
    #description of all the parameters used is located inside environment.py
    tf.keras.backend.clear_session()
    tf.keras.backend.reset_uids()

   
    if socket.gethostname() != "Orion":
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" # turns on xla and cpu xla
        tf.config.optimizer.set_jit(True)

    # set seed, so we can get the same results after rerunning several times
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    
    check_model_folders(save_folder, symbol)
    
    # model name to save, making it as unique as possible based on parameters
    model_name = (symbol + "-" + get_model_name(params))

    data, train, valid, test = load_data(symbol, params, end_date)

    model = create_model(params)

    logs_dir = "logs/" + get_time_string() + "-" + save_folder

    if params["SAVELOAD"]:
        checkpointer = ModelCheckpoint(directory_dict["model_dir"] + "/" + save_folder + "/" + model_name + ".h5", save_weights_only=True, save_best_only=True, verbose=1)
    else:    
        checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    
    if save_logs:
        tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch="200, 1200") 
    else:
        tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch=0)

    early_stop = EarlyStopping(patience=params["PATIENCE"])
    
    history = model.fit(train,
        batch_size=params["BATCH_SIZE"],
        epochs=params["EPOCHS"],
        verbose=2,
        validation_data=valid,
        callbacks = [tboard_callback, checkpointer, early_stop]   
    )

    epochs_used = len(history.history["loss"])
    #before testing, no shuffle
    if params["SAVELOAD"]:
        test_acc = valid_acc = train_acc = test_mae = valid_mae = train_mae = 0    
    else:    
        data, train, valid, test = load_data(symbol, params, end_date, False)

        model_path = os.path.join("results", model_name + ".h5")
        model.load_weights(model_path)

        if params["LOSS"] == "categorical_hinge":
            test_acc = valid_acc = train_acc = test_mae = valid_mae = train_mae = 0    
        else:
            test_mae, valid_mae, train_mae = get_all_maes(model, test, valid, train, data) 
            train_acc, valid_acc, test_acc = get_all_accuracies(model, data, params["LOOKUP_STEP"])


        delete_files_in_folder(directory_dict["results_dir"])
        os.rmdir(directory_dict["results_dir"])
        

    if not save_logs:
        delete_files_in_folder(logs_dir)
        os.rmdir(logs_dir)

    # data, model, test_acc, valid_acc, train_acc, test_mae, valid_mae, train_mae, 

    return epochs_used

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def nn_load_predict(symbol, current_date, params, model_name, save_folder):
    data, model = load_model_with_data(symbol, current_date, params, directory_dict["model_dir"], model_name, save_folder=save_folder)
    predicted_price = predict(model, data, params["N_STEPS"])

    return predicted_price

def ensemble_predictor(symbol, params, current_date):
    ensemb_count = 0
    ensemb_predict_list = []

    epochs_dict = {}
    df, train, valid, test = load_data(symbol, load_params, current_date, shuffle=False, to_print=False)
    df = df["df"]

    for predictor in params["ENSEMBLE"]:
        if "nn" in predictor:
            epochs_dict[predictor] = 0

    for predictor in params["ENSEMBLE"]:
        if predictor == "7MA":
            df["7MA"] = df.close.rolling(window=7).mean()
            predicted_price = np.float32(df["7MA"][len(df.close) - 1])
            ensemb_predict_list.append(predicted_price)
            
        elif predictor == "lin_reg":
            df["lin_reg"] = ta.LINEARREG(df.close, timeperiod=7)
            predicted_price = np.float32(df.lin_reg[len(df.close) - 1])
            ensemb_predict_list.append(predicted_price)

        elif "nn" in predictor:
            model_name = symbol + "-" + get_model_name(params[predictor])
            if params["TRADING"]:
                predicted_price = nn_load_predict(symbol, current_date, params[predictor], model_name)
            else:
                epochs_run = nn_train_save(symbol, current_date, params[predictor], params["SAVE_FOLDER"])
                epochs_dict[predictor] = epochs_run
                predicted_price = nn_load_predict(symbol, current_date, params[predictor], model_name, params["SAVE_FOLDER"])
            ensemb_predict_list.append(predicted_price)
        ensemb_count += 1

    final_prediction = mean(ensemb_predict_list)
    current_price = get_current_price(df)

    return final_prediction, current_price, epochs_dict


