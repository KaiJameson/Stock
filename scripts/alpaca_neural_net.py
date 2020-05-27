import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn import preprocessing
from time_functions import get_time_string
from environment import test_var, reports_directory, random_seed, error_file, back_test_days
from alpaca_nn_functions import load_data, create_model, predict, accuracy_score, plot_graph, get_accuracy, nn_report
from functions import delete_files_in_folder
import numpy as np
import pandas as pd
import os
import random
import sys
import time


def decision_neural_net(ticker, N_STEPS=300, LOOKUP_STEP=1, TEST_SIZE=0.2, 
    N_LAYERS=3, CELL=LSTM, UNITS=448, DROPOUT=0.3, BIDIRECTIONAL=True, LOSS="huber_loss",
    OPTIMIZER="adam", BATCH_SIZE=64, EPOCHS=2000):

    start_time = time.time()
    data, model, acc = make_neural_net(ticker, end_date=None, N_STEPS=300, LOOKUP_STEP=1, TEST_SIZE=0.2, 
    N_LAYERS=3, CELL=LSTM, UNITS=448, DROPOUT=0.3, BIDIRECTIONAL=True, LOSS="huber_loss",
    OPTIMIZER="adam", BATCH_SIZE=64, EPOCHS=2000)

    end_time = time.time()
    total_time = end_time - start_time
    percent = nn_report(ticker, total_time, model, data, acc, N_STEPS, LOOKUP_STEP)

    return percent, acc


def tuning_neural_net(ticker, end_date, N_STEPS=300, LOOKUP_STEP=1, TEST_SIZE=0.2, 
    N_LAYERS=3, CELL=LSTM, UNITS=448, DROPOUT=0.3, BIDIRECTIONAL=True, LOSS="huber_loss",
    OPTIMIZER="adam", BATCH_SIZE=64, EPOCHS=2000):
    
    data, model, acc = make_neural_net(ticker, end_date, N_STEPS, LOOKUP_STEP, TEST_SIZE, 
    N_LAYERS, CELL, UNITS, DROPOUT, BIDIRECTIONAL, LOSS,
    OPTIMIZER, BATCH_SIZE, EPOCHS)
    
    return acc

def make_neural_net(ticker, end_date=None, N_STEPS=300, LOOKUP_STEP=1, TEST_SIZE=0.2, 
    N_LAYERS=3, CELL=LSTM, UNITS=448, DROPOUT=0.3, BIDIRECTIONAL=True, LOSS="huber_loss",
    OPTIMIZER="adam", BATCH_SIZE=64, EPOCHS=2000):
    '''
    # N_STEPS = Window size or the sequence length
    # Lookup step = 1 is the next day
    # TEST_SIZE = 0.2 is 20%
    # N_LAYERS = how many hidden neural layers
    # CELL = type of cell
    # UNITS = number of neurons per layer
    # DROPOUT = % dropout
    # BIDIRECTIONAL = does it test backwards or not
    # LOSS = "huber_loss"
    # OPTIMIZER = "adam"
    # BATCH_SIZE
    # EPOCHS = how many times the machine trains
    '''
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
    data, train, test = load_data(
        ticker, N_STEPS, lookup_step=LOOKUP_STEP, 
        test_size=TEST_SIZE, batch_size=BATCH_SIZE, end_date=end_date
    )
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    history = model.fit(train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        use_multiprocessing=True
                        )

    model.save(os.path.join("results", model_name) + ".h5")
    #before testing, no shuffle
    data, train, test = load_data(
        ticker, N_STEPS, lookup_step=LOOKUP_STEP, 
        test_size=TEST_SIZE, shuffle=False, batch_size=BATCH_SIZE,
        end_date=end_date
    )


    # construct the model
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)
    
    delete_files_in_folder(results_folder)
    os.rmdir(results_folder)
    tf.keras.backend.clear_session()

    acc = get_accuracy(model, data, LOOKUP_STEP)

    return data, model, acc
