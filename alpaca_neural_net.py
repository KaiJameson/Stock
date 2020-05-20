import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from decision_tree import test_var

from alpaca_nn_functions import load_data, create_model, predict, accuracy_score, plot_graph, get_accuracy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random

import time
import threading
import logging


def main(symbols):
    #threads = list()
    for symbol in symbols:
        make_neural_net(symbol)
        #x = threading.Thread(target=make_neural_net, args=(symbol,))
        #threads.append(x)
        #x.start()


def deleteFiles(dirObject , dirPath):
    if dirObject.is_dir(follow_symlinks=False):
        name = os.fsdecode(dirObject.name)
        newDir = dirPath+"/"+name
        moreFiles = os.scandir(newDir)
        for file in moreFiles:
            if file.is_dir(follow_symlinks=False):
                deleteFiles(file, newDir)
                os.rmdir(newDir+"/"+os.fsdecode(file.name))
            else:
                os.remove(newDir+"/"+os.fsdecode(file.name))
        os.rmdir(newDir)
    else:
        os.remove(dirPath+"/"+os.fsdecode(dirObject.name))


def delete_files_in_folder(directory):
    try:
        files = os.scandir(directory)
        for file in files:
            deleteFiles(file, directory)
    except:
        print("problem with removing files in " + str(directory))


def make_neural_net(ticker, N_STEPS=300, LOOKUP_STEP=1, TEST_SIZE=0.2, 
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
    seed = 314
    # set seed, so we can get the same results after rerunning several times
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    start_time = time.time()
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
    data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE)
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        verbose=1)

    model.save(os.path.join("results", model_name) + ".h5")
    #before testing, no shuffle
    data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, shuffle=False)

    # construct the model
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)
    # evaluate the model
    mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # predict the future price
    future_price = predict(model, data, N_STEPS)
    #print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    delete_files_in_folder(results_folder)
    os.rmdir(results_folder)
    end_time = time.time()
    total_time = end_time - start_time
    total_minutes = total_time / 60
    curr_price = plot_graph(model, data, ticker=ticker)
    file_name = 'reports/' + ticker +'_' + test_var + '.txt'
    f = open(file_name, 'a')
    f.write('Total time to run was: ' + str(total_minutes) + '\n')
    f.write('The price at run time was: ' + str(curr_price) + '\n')
    f.write('The predicted price for tomorrow is ' + str(future_price) + '\n')
    if curr_price < future_price:
        f.write('i would buy this stock\n')
    elif curr_price > future_price:
        f.write('i would sell this stock\n')
    percent = future_price / curr_price
    acc = get_accuracy(model, data, LOOKUP_STEP)
    f.write(str(LOOKUP_STEP) + ":" + "Accuracy Score:" + str(acc) + '\n')
    f.close()
    return percent, acc


if __name__== '__main__':
    #symbols = ['TSLA', 'PENN', 'ZOM', "AHPI"]
    symbols = ['WMT','AAPL', 'WTRH', 'MVIS']
    main(symbols)
    #make_neural_net('ZOM')
