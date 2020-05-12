import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from collections import deque


from alpaca_nn_functions import load_data, create_model, predict, accuracy_score, plot_graph, get_accuracy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random

import time

seed = 314
# set seed, so we can get the same results after rerunning several times
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
start_time = time.time()


# Window size or the sequence length
N_STEPS = 100
# Lookup step, 1 is the next day
LOOKUP_STEP = 1
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["open", "low", "high", "close", "volume", "mid"]
# date now
date_now = time.strftime("%Y-%m-%d")
### model parameters
N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False
### training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 1000
# Apple stock market
ticker = "AAPL"
#ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"
# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
# if not os.path.isdir("logs"):
#     os.mkdir("logs")
# if not os.path.isdir("data"):
#     os.mkdir("data")
# load the data
data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)
#print('here is the data[X_test]')
#print(data['X_test'])
#print('here is the data[y_test]')
#print(data['y_test'])
# save the dataframe
#data["df"].to_csv(ticker_data_filename)

# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
#print('here is the model')
#print(model)
# some tensorflow callbacks
#checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
#tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
#                    callbacks=[checkpointer, tensorboard],

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    verbose=1)

model.save(os.path.join("results", model_name) + ".h5")


#before testing, no shuffle
data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS, shuffle=False)

# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)
# evaluate the model
mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
#print('mae', mae)
#print('mse', mse)
# calculate the mean absolute error (inverse scaling)
#mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
#TODO: THERE IS A BIG PROBLEM WITH THE ABOVE LINE, RESHAPE NO WORK ON FLOAT
#print("Mean Absolute Error:", mean_absolute_error)



# predict the future price
future_price = predict(model, data, N_STEPS)
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")




end_time = time.time()
total_time = end_time - start_time
total_minutes = total_time / 60
print('total minutes is', total_minutes)
plot_graph(model, data)


print(str(LOOKUP_STEP) + ":", "Accuracy Score:", get_accuracy(model, data, LOOKUP_STEP))

