from config.silen_ten import silence_tensorflow
silence_tensorflow()
from functions.functions import delete_files_in_folder, check_model_folders, get_model_name
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from config.environ import directory_dict, random_seed, save_logs, defaults
from tensorflow.keras.layers import LSTM
from functions.time_functs import get_time_string, get_past_date_string
from functions.functions import get_model_name, layer_name_converter, sr2
from functions.paca_model_functs import create_model, get_accuracy, get_all_accuracies, get_current_price, predict, return_real_predict
from functions.io_functs import save_prediction, load_saved_predictions
from scipy.signal import savgol_filter
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from statistics import mean
import pandas as pd
import talib as ta
import numpy as np
import socket
import gc
import random
import os

def nn_train_save(symbol, params=defaults, end_date=None, predictor="nn1", data_dict={}):
    #description of all the parameters used is located inside environment.py
    gc.collect()
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
   
    model_name = (symbol + "-" + get_model_name(nn_params))

    
    model = create_model(nn_params)

    logs_dir = "logs/" + get_time_string() + "-" + params["SAVE_FOLDER"]

    checkpointer = ModelCheckpoint(directory_dict["model"] + "/" + params["SAVE_FOLDER"] + "/" 
        + model_name + ".h5", save_weights_only=True, save_best_only=True, verbose=1)
    
    
    # checkpointer = ModelCheckpoint(directory_dict["model"] + "/" + params["SAVE_FOLDER"] + "/" 
    #     + model_name + ".h5", save_weights_only=True, verbose=1)

    if save_logs:
        tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch="200, 1200") 
    else:
        tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch=0)

    early_stop = EarlyStopping(patience=nn_params["PATIENCE"])
    
    history = model.fit(data_dict["train"],
        batch_size=nn_params["BATCH_SIZE"],
        epochs=nn_params["EPOCHS"],
        verbose=2,
        validation_data=data_dict["valid"],
        callbacks = [tboard_callback, checkpointer, early_stop]   
        # callbacks = [checkpointer, tboard_callback]
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

def nn_load_predict(symbol, params, predictor, data_dict, to_print=False):
    model = create_model(params[predictor])
    model.load_weights(directory_dict["model"] + "/" + params["SAVE_FOLDER"] + "/" + 
        symbol + "-" + get_model_name(params[predictor]) + ".h5")
    predicted_price = predict(model, data_dict["result"], params[predictor]["N_STEPS"], params[predictor]["TEST_VAR"],
        layer=params[predictor]["LAYERS"][0])

    return predicted_price

def ensemble_predictor(symbol, params, current_date, data_dict, df):
    ensemb_predict_list = []
    epochs_dict = {}
    
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
            
        elif predictor == "lin_reg":
            df["lin_reg"] = ta.LINEARREG(df.c, timeperiod=7)
            predicted_price = np.float32(df.lin_reg[len(df.c) - 1])

        elif predictor == "sav_gol":
            df["sc"] = savgol_filter(df.c, 7, 3)
            predicted_price = np.float32(df.sc[len(df.c) - 1])

        elif predictor == "EMA":
            df["EMA"] = ta.EMA(df.c, timeperiod=5)
            predicted_price = np.float32(df["EMA"][len(df.c) - 1])

        #TODO see if we can implement the tech_dict to resolve the above in one section

        elif "DTREE" in predictor:
            tree = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
                min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"])
            tree.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
            # imps = permutation_importance(tree, data_dict[predictor]["X_train"],
            #     data_dict[predictor]["y_train"])["importances_mean"]
            # for i,feature in enumerate(params[predictor]["FEATURE_COLUMNS"]):
            #     print(f"{feature} has importance of {imps[i]}")
            tree_pred = tree.predict(data_dict[predictor]["X_test"])
            scale =data_dict[predictor]["column_scaler"]["future"]
            tree_pred = np.array(tree_pred)
            tree_pred = tree_pred.reshape(1, -1)
            predicted_price = np.float32(scale.inverse_transform(tree_pred)[-1][-1])

        elif "RFORE" in predictor:
            fore = RandomForestRegressor(n_estimators=params[predictor]["N_ESTIMATORS"],
                max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], n_jobs=-1)
            fore.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
            fore_pred = fore.predict(data_dict[predictor]["X_test"])
            scale = data_dict[predictor]["column_scaler"]["future"]
            fore_pred = np.array(fore_pred)
            fore_pred = fore_pred.reshape(1, -1)
            predicted_price = np.float32(scale.inverse_transform(fore_pred)[-1][-1])

        elif "KNN" in predictor:
            knn = KNeighborsRegressor(n_neighbors=params[predictor]["N_NEIGHBORS"], n_jobs=-1)
            knn.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
            knn_pred = knn.predict(data_dict[predictor]["X_test"])
            scale = data_dict[predictor]["column_scaler"]["future"]
            knn_pred = np.array(knn_pred)
            knn_pred = knn_pred.reshape(1, -1)
            predicted_price = np.float32(scale.inverse_transform(knn_pred)[-1][-1])

        elif "ADA" in predictor:
            base = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
                min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"])
            ada = AdaBoostRegressor(base_estimator=base, n_estimators=params[predictor]["N_ESTIMATORS"])
            ada.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
            # imps = permutation_importance(ada, data_dict[predictor]["X_train"],
            #     data_dict[predictor]["y_train"])["importances_mean"]
            # for i,feature in enumerate(params[predictor]["FEATURE_COLUMNS"]):
            #     print(f"{feature} has importance of {imps[i]}")
            ada_pred = ada.predict(data_dict[predictor]["X_test"])
            scale = data_dict[predictor]["column_scaler"]["future"]
            ada_pred = np.array(ada_pred)
            ada_pred = ada_pred.reshape(1, -1)
            predicted_price = np.float32(scale.inverse_transform(ada_pred)[-1][-1])

        elif "nn" in predictor:
            if params["TRADING"]:
                predicted_price = nn_load_predict(symbol, params, predictor, data_dict[predictor])
            else:
                if load_saved_predictions(symbol, params, current_date, predictor):
                    predicted_price, epochs_run = load_saved_predictions(symbol, params, current_date, predictor)
                    epochs_dict[predictor] = epochs_run
                else:
                    epochs_run = nn_train_save(symbol, params, current_date, predictor, data_dict[predictor])
                    epochs_dict[predictor] = epochs_run
                    predicted_price = nn_load_predict(symbol, params, predictor, data_dict[predictor])
                    save_prediction(symbol, params, current_date, predictor, predicted_price, epochs_run)
        ensemb_predict_list.append(np.float32(predicted_price))

    print(f"Ensemble prediction list: {ensemb_predict_list}")
    final_prediction = mean(ensemb_predict_list)
    print(f"The final prediction: {sr2(final_prediction)}")
    current_price = get_current_price(df)

    return final_prediction, current_price, epochs_dict


def ensemble_accuracy(symbol, params, current_date, classification=False):
    for predictor in params:
        if "nn" in predictor:
            pass
            # data, model = load_model_with_data(symbol, current_date, params, predictor)
            model = {}
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
