import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn import preprocessing
from time_functions import get_time_string
from environment import (test_var, reports_directory, model_saveload_directory, random_seed, error_file, back_test_days, 
save_logs)
from alpaca_nn_functions import (load_data, create_model, predict, accuracy_score, plot_graph, 
get_accuracy, nn_report, return_real_predict, get_all_accuracies, get_all_maes)
from functions import delete_files_in_folder
from time_functions import get_time_string
from datetime import datetime
from environment import defaults
import numpy as np
import pandas as pd
import os
import random
import sys
import time


def decision_neural_net(
    ticker, 
    N_STEPS=defaults["N_STEPS"], 
    LOOKUP_STEP=defaults["LOOKUP_STEP"], 
    TEST_SIZE=defaults["TEST_SIZE"], 
    N_LAYERS=defaults["N_LAYERS"], 
    CELL=defaults["CELL"], 
    UNITS=defaults["UNITS"], 
    DROPOUT=defaults["DROPOUT"], 
    BIDIRECTIONAL=defaults["BIDIRECTIONAL"], 
    LOSS=defaults["LOSS"],
    OPTIMIZER=defaults["OPTIMIZER"], 
    BATCH_SIZE=defaults["BATCH_SIZE"], 
    EPOCHS=defaults["EPOCHS"],
    PATIENCE=defaults["PATIENCE"],
    SAVELOAD=defaults["SAVELOAD"],
    LIMIT=defaults["LIMIT"],
    FEATURE_COLUMNS=defaults["FEATURE_COLUMNS"]):
    #description of these parameters located inside environment.py

    start_time = time.time()
    data, model, test_acc, valid_acc, train_acc, test_mae, valid_mae, train_mae, epochs_used = make_neural_net(
        ticker, None, N_STEPS, LOOKUP_STEP, TEST_SIZE, N_LAYERS, CELL, UNITS, DROPOUT, 
        BIDIRECTIONAL, LOSS, OPTIMIZER, BATCH_SIZE, EPOCHS, PATIENCE, SAVELOAD, LIMIT, FEATURE_COLUMNS
    )

    end_time = time.time()
    total_time = end_time - start_time
    percent = nn_report(ticker, total_time, model, data, test_acc, valid_acc, train_acc, N_STEPS)

    return percent, test_acc


def tuning_neural_net(ticker, end_date, 
    N_STEPS=defaults["N_STEPS"], 
    LOOKUP_STEP=defaults["LOOKUP_STEP"], 
    TEST_SIZE=defaults["TEST_SIZE"], 
    N_LAYERS=defaults["N_LAYERS"], 
    CELL=defaults["CELL"], 
    UNITS=defaults["UNITS"], 
    DROPOUT=defaults["DROPOUT"], 
    BIDIRECTIONAL=defaults["BIDIRECTIONAL"], 
    LOSS=defaults["LOSS"],
    OPTIMIZER=defaults["OPTIMIZER"], 
    BATCH_SIZE=defaults["BATCH_SIZE"], 
    EPOCHS=defaults["EPOCHS"],
    PATIENCE=defaults["PATIENCE"],
    SAVELOAD=defaults["SAVELOAD"],
    LIMIT=defaults["LIMIT"],
    FEATURE_COLUMNS=defaults["FEATURE_COLUMNS"]):
    #description of these parameters located inside environment.py
    
    data, model, test_acc, valid_acc, train_acc, test_mae, valid_mae, train_mae, epochs_used = make_neural_net(
        ticker, end_date, N_STEPS, LOOKUP_STEP, TEST_SIZE, N_LAYERS, CELL, UNITS, 
        DROPOUT, BIDIRECTIONAL, LOSS, OPTIMIZER, BATCH_SIZE, EPOCHS, PATIENCE, SAVELOAD, LIMIT, 
        FEATURE_COLUMNS
    )
    
    return test_acc, valid_acc, train_acc, test_mae, valid_mae, train_mae, epochs_used

def saveload_neural_net(ticker, end_date=None, 
    N_STEPS=defaults["N_STEPS"], 
    LOOKUP_STEP=defaults["LOOKUP_STEP"], 
    TEST_SIZE=defaults["TEST_SIZE"], 
    N_LAYERS=defaults["N_LAYERS"], 
    CELL=defaults["CELL"], 
    UNITS=defaults["UNITS"], 
    DROPOUT=defaults["DROPOUT"], 
    BIDIRECTIONAL=defaults["BIDIRECTIONAL"], 
    LOSS=defaults["LOSS"],
    OPTIMIZER=defaults["OPTIMIZER"], 
    BATCH_SIZE=defaults["BATCH_SIZE"], 
    EPOCHS=defaults["EPOCHS"],
    PATIENCE=defaults["PATIENCE"],
    SAVELOAD=defaults["SAVELOAD"],
    LIMIT=defaults["LIMIT"],
    FEATURE_COLUMNS=defaults["FEATURE_COLUMNS"]):

    data, model, test_acc, valid_acc, train_acc, test_mae, valid_mae, train_mae, epochs_used = make_neural_net(
        ticker, end_date, N_STEPS, LOOKUP_STEP, TEST_SIZE, N_LAYERS, CELL, UNITS, 
        DROPOUT, BIDIRECTIONAL, LOSS, OPTIMIZER, BATCH_SIZE, EPOCHS, PATIENCE, SAVELOAD, LIMIT, 
        FEATURE_COLUMNS
    )


def make_neural_net(ticker, end_date, N_STEPS, LOOKUP_STEP, TEST_SIZE, N_LAYERS, CELL, UNITS, 
        DROPOUT, BIDIRECTIONAL, LOSS, OPTIMIZER, BATCH_SIZE, EPOCHS, PATIENCE, SAVELOAD, LIMIT, 
        FEATURE_COLUMNS):
    #description of these parameters located inside environment.py

    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True)

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)

    # set seed, so we can get the same results after rerunning several times
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    
    # date now
    date_now = time.strftime("%Y-%m-%d")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"{ticker}-{FEATURE_COLUMNS}-limit-{LIMIT}-{CELL.__name__}-n_step-{N_STEPS}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"
    # create these folders if they do not exist
    results_folder = "results"
    if not os.path.isdir(results_folder):
       os.mkdir(results_folder)
    data, train, valid, test = load_data(ticker, end_date, N_STEPS, BATCH_SIZE, LIMIT, FEATURE_COLUMNS)

    model = create_model(N_STEPS, UNITS, CELL, N_LAYERS, DROPOUT, LOSS, OPTIMIZER, BIDIRECTIONAL)

    logs = "logs/" + get_time_string()

    if SAVELOAD:
        checkpointer = ModelCheckpoint(model_saveload_directory + "/" + ticker + ".h5", save_weights_only=True, save_best_only=True, verbose=1)
    else:    
        checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    
    if save_logs:
        tboard_callback = TensorBoard(log_dir=logs, profile_batch="100,200") 
    else:
        tboard_callback = TensorBoard(log_dir=logs, profile_batch=0)

    early_stop = EarlyStopping(patience=PATIENCE)
    
    
    history = model.fit(train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        use_multiprocessing=True,
                        validation_data=valid,
                        callbacks = [tboard_callback, checkpointer, early_stop]   
                        )

    epochs_used = len(history.history["loss"])

    #before testing, no shuffle
    if SAVELOAD:
        test_acc = valid_acc = train_acc = test_mae = valid_mae = train_mae = 0    
    else:    
        data, train, valid, test = load_data( ticker, end_date, N_STEPS, BATCH_SIZE, 
            LIMIT, FEATURE_COLUMNS, False
        )

        model_path = os.path.join("results", model_name + ".h5")
        model.load_weights(model_path)

        test_mae, valid_mae, train_mae = get_all_maes(model, test, valid, train, data) 

        delete_files_in_folder(results_folder)
        os.rmdir(results_folder)
        
        train_acc, valid_acc, test_acc = get_all_accuracies(model, data, LOOKUP_STEP)

    if not save_logs:
        delete_files_in_folder(logs)

    return data, model, test_acc, valid_acc, train_acc, test_mae, valid_mae, train_mae, epochs_used
