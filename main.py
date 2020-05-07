import pandas as pd
from sklearn import preprocessing
import numpy as np
import datetime as dt
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
from DataGeneratorSeq import DataGeneratorSeq
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf


time_period = 50
csv_location = 'EOD-WMT.csv'
cols=['Date','Open','High','Low','Close']
# csv_location = 'OPEC-ORB.csv'
# cols = ['Date', 'Value']
split_location = 800


def standard_avg(train_data, df):
    window_size = 10
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size, N):
        date = df.loc[pred_idx, 'Date']
        # print('printing date')
        # print(date)
        std_avg_predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx]) ** 2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f' % (0.5 * np.mean(mse_errors)))
    plt.figure(figsize=(12, 6))
    plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
    plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()



def exp_mov_avg(train_data, df):
    N = train_data.size

    run_avg_predictions = []
    run_avg_x = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1, N):
        running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)
        # print(df.loc[pred_idx, 'Date'])
        run_avg_x.append(df.loc[pred_idx, 'Date'])

    print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))
    plt.figure(figsize=(12, 6))
    plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
    plt.plot(range(0, N), run_avg_predictions, color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()


df = pd.read_csv(csv_location, usecols=cols)
df = df.sort_values('Date')
print(df.head())
# plt.figure(figsize = (12,6))
# plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
# plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Mid Price', fontsize=18)
# plt.show()
# First calculate the mid prices from the highest and lowest

high_prices = df.loc[:,'High'].to_numpy()
low_prices = df.loc[:,'Low'].to_numpy()
mid_prices = (high_prices+low_prices)/2.0
# mid_prices = df.loc[:,'Value'].to_numpy()
train_data = mid_prices[:split_location]
test_data = mid_prices[split_location:]
# Scale the data to be between 0 and 1
# When scaling remember! You normalize both test and train data with respect to training data
# Because you are not supposed to have access to test data
scaler = preprocessing.MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)
# Train the Scaler with training data and smooth data
smoothing_window_size = (split_location-100) // 4
for di in range(0, split_location-100, smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)
# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(split_location):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)

#standard_avg(train_data, df)
exp_mov_avg(train_data, df)

dg = DataGeneratorSeq(train_data,5,5)
u_data, u_labels = dg.unroll_batches()

for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ',dat )
    print('\n\tOutput:',lbl)

D = 1 # Dimensionality of the data. Since your data is 1-D this would be 1
num_unrollings = 50 # Number of time steps you look into the future.
batch_size = 500 # Number of samples in a batch
num_nodes = [200,200,150] # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes) # number of layers
dropout = 0.2 # dropout amount

ops.reset_default_graph() # This is important in case you run this multiple times

# Input data.
train_inputs, train_outputs = [],[]

# You unroll the input over time defining placeholders for each time step
for ui in range(num_unrollings):
    train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
    train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))

lstm_cells = [
    tf.distribute.rnn.LSTMCell(num_units=num_nodes[li],
                            state_is_tuple=True,
                            initializer= tf.distribute.layers.xavier_initializer()
                           )
 for li in range(n_layers)]

drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
    lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
) for lstm in lstm_cells]
drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

w = tf.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))


