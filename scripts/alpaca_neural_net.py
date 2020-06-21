import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn import preprocessing
from time_functions import get_time_string
from environment import test_var, reports_directory, random_seed, error_file, back_test_days, save_logs
from alpaca_nn_functions import load_data, create_model, predict, accuracy_score, plot_graph, get_accuracy, nn_report
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
    N_STEPS=defaults['N_STEPS'], 
    LOOKUP_STEP=defaults['LOOKUP_STEP'], 
    TEST_SIZE=defaults['TEST_SIZE'], 
    N_LAYERS=defaults['N_LAYERS'], 
    CELL=defaults['CELL'], 
    UNITS=defaults['UNITS'], 
    DROPOUT=defaults['DROPOUT'], 
    BIDIRECTIONAL=defaults['BIDIRECTIONAL'], 
    LOSS=defaults['LOSS'],
    OPTIMIZER=defaults['OPTIMIZER'], 
    BATCH_SIZE=defaults['BATCH_SIZE'], 
    EPOCHS=defaults['EPOCHS'],
    PATIENCE=defaults['PATIENCE']):
#description of these parameters located inside environment.py

    start_time = time.time()
    data, model, acc, mae = make_neural_net(
        ticker, N_STEPS=N_STEPS, LOOKUP_STEP=LOOKUP_STEP, TEST_SIZE=TEST_SIZE, 
        N_LAYERS=N_LAYERS, CELL=CELL, UNITS=UNITS, DROPOUT=DROPOUT, 
        BIDIRECTIONAL=BIDIRECTIONAL, LOSS=LOSS, OPTIMIZER=OPTIMIZER, 
        BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS, PATIENCE=PATIENCE
    )

    end_time = time.time()
    total_time = end_time - start_time
    percent = nn_report(ticker, total_time, model, data, acc, mae, N_STEPS, LOOKUP_STEP)

    return percent, acc


def tuning_neural_net(ticker, end_date, 
    N_STEPS=defaults['N_STEPS'], 
    LOOKUP_STEP=defaults['LOOKUP_STEP'], 
    TEST_SIZE=defaults['TEST_SIZE'], 
    N_LAYERS=defaults['N_LAYERS'], 
    CELL=defaults['CELL'], 
    UNITS=defaults['UNITS'], 
    DROPOUT=defaults['DROPOUT'], 
    BIDIRECTIONAL=defaults['BIDIRECTIONAL'], 
    LOSS=defaults['LOSS'],
    OPTIMIZER=defaults['OPTIMIZER'], 
    BATCH_SIZE=defaults['BATCH_SIZE'], 
    EPOCHS=defaults['EPOCHS'],
    PATIENCE=defaults['PATIENCE']):
#description of these parameters located inside environment.py
    
    data, model, acc, mae = make_neural_net(
        ticker, end_date=end_date, 
        N_STEPS=N_STEPS, LOOKUP_STEP=LOOKUP_STEP, TEST_SIZE=TEST_SIZE, 
        N_LAYERS=N_LAYERS, CELL=CELL, UNITS=UNITS, DROPOUT=DROPOUT, 
        BIDIRECTIONAL=BIDIRECTIONAL, LOSS=LOSS, OPTIMIZER=OPTIMIZER, 
        BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS, PATIENCE=PATIENCE
    )
    
    return acc, mae

def make_neural_net(ticker, end_date=None, 
    N_STEPS=defaults['N_STEPS'], 
    LOOKUP_STEP=defaults['LOOKUP_STEP'], 
    TEST_SIZE=defaults['TEST_SIZE'], 
    N_LAYERS=defaults['N_LAYERS'], 
    CELL=defaults['CELL'], 
    UNITS=defaults['UNITS'], 
    DROPOUT=defaults['DROPOUT'], 
    BIDIRECTIONAL=defaults['BIDIRECTIONAL'], 
    LOSS=defaults['LOSS'],
    OPTIMIZER=defaults['OPTIMIZER'], 
    BATCH_SIZE=defaults['BATCH_SIZE'], 
    EPOCHS=defaults['EPOCHS'],
    PATIENCE=defaults['PATIENCE']):
#description of these parameters located inside environment.py

    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    # set seed, so we can get the same results after rerunning several times
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    
    # date now
    date_now = time.strftime("%Y-%m-%d")
    #ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"
    # create these folders if they does not exist
    results_folder = 'results'
    if not os.path.isdir(results_folder):
       os.mkdir(results_folder)
    data, train, valid, test = load_data(
        ticker, N_STEPS, lookup_step=LOOKUP_STEP, 
        test_size=TEST_SIZE, batch_size=BATCH_SIZE, end_date=end_date
    )
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    logs = "logs/" + get_time_string()

    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    
    if save_logs:
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, profile_batch='100,200') 
    else:
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, profile_batch=0)

    early_stop = tf.keras.callbacks.EarlyStopping(patience=PATIENCE)
    
    
    history = model.fit(train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        use_multiprocessing=True,
                        validation_data=valid,
                        callbacks = [tboard_callback, checkpointer, early_stop]   
                        )

    
    #before testing, no shuffle
    data, train, valid, test = load_data(
        ticker, N_STEPS, lookup_step=LOOKUP_STEP, 
        test_size=TEST_SIZE, shuffle=False, batch_size=BATCH_SIZE,
        end_date=end_date
    )

    model_path = os.path.join("results", model_name + ".h5")
    model.load_weights(model_path)

    mse, mae = model.evaluate(test, verbose=0)
    mae = data["column_scaler"][test_var].inverse_transform([[mae]])[0][0]
    
    delete_files_in_folder(results_folder)
    os.rmdir(results_folder)
    

    acc = get_accuracy(model, data, LOOKUP_STEP)

    return data, model, acc, mae
