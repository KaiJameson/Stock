from pandas import DataFrame
from sklearn.neural_network import MLPRegressor
from config.environ import *
from config.symbols import *
from config.api_key import *
from functions.tuner import *
from functions.paca_model import *
from functions.data_load import *
from functions.io import *
from functions.functions import *
from functions.time import *
from functions.technical_indicators import *
from functions.volitility import *
from functions.all_2D_models import *
from paca_model import *
from load_run import *
from tuner import tuning
from statistics import mean
from scipy.signal import cwt
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error
from tensorflow_addons.layers import ESN
from iexfinance.stocks import Stock, get_historical_data
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from mlens.ensemble import SuperLearner
from mlens.model_selection import Evaluator
from scipy.io import loadmat
from collections import Counter
from multiprocessing.pool import Pool
import keras_tuner as kt
import nltk
import math
import requests
import scipy
import keras
import copy


def backtest_comparator(start_day, end_day, comparator, run_days):
    load_save_symbols = ["AGYS", "AMKR", "BG","BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
        "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]

    over_all = {i:[0.0, 0.0, 0.0]
    for i in range(start_day, end_day)}

    
    for symbol in load_save_symbols:
        print(symbol, flush=True)
        for i in range(start_day, end_day):
            data, train, valid, test = load_3D_data(symbol, params=defaults["nn1"], end_date=None, shuffle=False, to_print=False)
            if comparator == "7MA":
                avg = MA_comparator(data, i, run_days)
            elif comparator == "lin_reg":
                avg = lin_reg_comparator(data, i, run_days)
            elif comparator == "EMA":
                avg = EMA_comparator(data, i, run_days)
            elif comparator == "TSF":
                avg = TSF_comparator(data, i, run_days)
            elif comparator == "sav_gol":
                if i == 1 or i == 3:
                    continue
                elif i % 2 == 0:
                    continue
                else:
                    avg = sav_gol_comparator(data, i, 4, run_days)


            over_all[i][0] += float(avg[0])
            over_all[i][1] += float(avg[1])
            over_all[i][2] += float(avg[2])

    print(f"~~~  {comparator}  ~~~")
    for j in range(start_day, end_day):
        print(f"{j}", end="")
        for metric in over_all[j]:
            print(f" {round(metric / len(load_save_symbols), 2)} ", end="")
        print()



if __name__ == "__main__":

    params = {
        "ENSEMBLE": ["MLENS1", "RFORE1"],
        "TRADING": False,
        "SAVE_FOLDER": "tune4",
        "nn1" : { 
            "N_STEPS": 100,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 0.2,
            "LAYERS": [(256, Dense), (256, Dense), (256, Dense), (256, Dense)],
            "SHUFFLE": True,
            "DROPOUT": .4,
            "BIDIRECTIONAL": False,
            "LOSS": "huber_loss",
            "OPTIMIZER": "adam",
            "BATCH_SIZE": 1024,
            "EPOCHS": 2000,
            "PATIENCE": 200,
            "SAVELOAD": True,
            "LIMIT": 4000,
            "FEATURE_COLUMNS": ["c", "o", "l", "h", "m", "v"],
            "TEST_VAR": "c",
            "SAVE_PRED": {}
        },
        "DTREE1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v"],
            "MAX_DEPTH": 5,
            "MIN_SAMP_LEAF": 1,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "RFORE1" : {
            "FEATURE_COLUMNS": ["c", "vwap"],
            "N_ESTIMATORS": 1000,
            "MAX_DEPTH": 10000,
            "MIN_SAMP_LEAF": 1,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "KNN1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "testing"],
            "N_NEIGHBORS": 5,
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "ADA1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v", "tc", "vwap"],
            "N_ESTIMATORS": 100,
            "MAX_DEPTH": 10000,
            "MIN_SAMP_LEAF": 1,
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "XGB1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v", "tc", "vwap"],
            "N_ESTIMATORS": 100,
            "MAX_DEPTH": 1000,
            "MAX_LEAVES": 1000,
            "GAMMA": 0.0,
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "MLP1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v", "tc", "vwap"],
            "LAYERS": (1),
            "EARLY_STOP": True,
            "VALIDATION_FRACTION": .2,
            "PATIENCE": 69,
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "MLENS1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v", "tc", "vwap"],
            "LAYERS": [["MLP1", "RFORE1"], ["ADA1", "KNN1"], ["XGB1", "BAGREG1"], ["DTREE1", "XTREE1"]],
            "META_EST": "MLP",
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c",
            "DTREE1" : {
                "MAX_DEPTH": 1000,
                "MIN_SAMP_LEAF": 5,
            },
            "XTREE1" : {
                "N_ESTIMATORS": 10,
                "MAX_DEPTH": 1000,
                "MIN_SAMP_LEAF": 1,
            },
            "RFORE1" : {
                "N_ESTIMATORS": 10,
                "MAX_DEPTH": 1000,
                "MIN_SAMP_LEAF": 1,
            },
            "KNN1" : {
                "N_NEIGHBORS": 5,
                "WEIGHTS": "distance"
            },
            "ADA1" : {
                "N_ESTIMATORS": 100,
                "MAX_DEPTH": 1000,
                "MIN_SAMP_LEAF": 1,
            },
            "XGB1" : {
                    "N_ESTIMATORS": 50,
                    "MAX_DEPTH": 10,
                    "MAX_LEAVES": 1000,
            },
            "XGB2" : {
                    "N_ESTIMATORS": 100,
                    "MAX_DEPTH": 10,
                    "MAX_LEAVES": 1000,
            },
            "BAGREG1" : {
                "N_ESTIMATORS": 10,
                "MAX_DEPTH": 1000,
                "MIN_SAMP_LEAF": 1,
            },
            "MLP1" : { 
                "LAYERS": (10), 
                "EARLY_STOP": True,
                "VALIDATION_FRACTION": .2,
                "PATIENCE": 5,
            },
            "MLP2" : { 
                "LAYERS": (10, 10), 
                "EARLY_STOP": True,
                "VALIDATION_FRACTION": .2,
                "PATIENCE": 5,
            },
            "MLP3" : { 
                "LAYERS": (20, 10, 10), 
                "EARLY_STOP": True,
                "VALIDATION_FRACTION": .2,
                "PATIENCE": 5,
            },
        },
        "TRANS1" :{
            "FEATURE_COLUMNS": ["pc.o", "pc.l", "pc.h", "pc.c", "pc.v"],
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c",

        },
        "LIMIT": 4000,
    }

    params2 = {
         "nn1" : { 
            "N_STEPS": 100,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 0.2,
            "LAYERS": [(256, Dense), (256, Dense), (256, Dense), (256, Dense)],
            "SHUFFLE": True,
            "DROPOUT": .4,
            "BIDIRECTIONAL": False,
            "LOSS": "huber_loss",
            "OPTIMIZER": "adam",
            "BATCH_SIZE": 1024,
            "EPOCHS": 2000,
            "PATIENCE": 200,
            "SAVELOAD": True,
            "LIMIT": 4000,
            "FEATURE_COLUMNS": ["c", "es.c"],
            "TEST_VAR": "c",
            "SAVE_PRED": {}
        },

    }
 


    api = get_api()

    s = time.perf_counter()
    configure_gpu()
    # TRANSFORMER SECTION
    # import numpy as np
    # import pandas as pd
    # import os, datetime
    # import tensorflow as tf
    # from tensorflow.keras.models import *
    # from tensorflow.keras.layers import *
    # print('Tensorflow version: {}'.format(tf.__version__))

    # import matplotlib.pyplot as plt
    # plt.style.use('seaborn')

    # batch_size = 32
    # seq_len = 128

    # d_k = 256
    # d_v = 256
    # n_heads = 12
    # ff_dim = 256

    # symbol = "AGYS"

    # df = get_proper_df(symbol, 4000, "V2")
    # df = modify_dataframe(params["TRANS1"]["FEATURE_COLUMNS"], df, True)
    # df = df.replace(0, 0.000000001)
    # #df =  df.replace(to_replace=0, method='ffill', inplace=True) 

    # df = df.reset_index()
    # df = df.drop(columns=["c", "timestamp"])
    # df = df.dropna(how='any', axis=0)


    # '''Create indexes to split dataset'''

    # times = sorted(df.index.values)
    # last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
    # last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

    # ###############################################################################
    # '''Normalize price columns'''
    # #
    # min_return = min(df[(df.index < last_20pct)][["pc.o", "pc.l", "pc.h", "pc.c",]].min(axis=0))
    # max_return = max(df[(df.index < last_20pct)][["pc.o", "pc.l", "pc.h", "pc.c",]].max(axis=0))

    # # Min-max normalize price columns (0-1 range)
    # df['pc.o'] = (df['pc.o'] - min_return) / (max_return - min_return)
    # df['pc.h'] = (df['pc.h'] - min_return) / (max_return - min_return)
    # df['pc.l'] = (df['pc.l'] - min_return) / (max_return - min_return)
    # df['pc.c'] = (df['pc.c'] - min_return) / (max_return - min_return)

    # ###############################################################################
    # '''Normalize volume column'''

    # min_volume = df[(df.index < last_20pct)]['pc.v'].min(axis=0)
    # max_volume = df[(df.index < last_20pct)]['pc.v'].max(axis=0)

    # # Min-max normalize volume columns (0-1 range)
    # df['pc.v'] = (df['pc.v'] - min_volume) / (max_volume - min_volume)

    # ###############################################################################
    # '''Create training, validation and test split'''

    # df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    # df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    # df_test = df[(df.index >= last_10pct)]

    # print(df_train)

    # # Remove date column
    # # df_train.drop(columns=['Date'], inplace=True)
    # # df_val.drop(columns=['Date'], inplace=True)
    # # df_test.drop(columns=['Date'], inplace=True)

    # # Convert pandas columns into arrays
    # train_data = df_train.values
    # # print(train_data)
    # val_data = df_val.values
    # test_data = df_test.values
    # print('Training data shape: {}'.format(train_data.shape))
    # print('Validation data shape: {}'.format(val_data.shape))
    # print('Test data shape: {}'.format(test_data.shape))

    # # Training data
    # X_train, y_train = [], []
    # for i in range(seq_len, len(train_data)):
    #     X_train.append(train_data[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
    #     y_train.append(train_data[:, 3][i]) #Value of 4th column (Close Price) of df-row 128+1
    # X_train, y_train = np.array(X_train), np.array(y_train)

    # ###############################################################################

    # # Validation data
    # X_val, y_val = [], []
    # for i in range(seq_len, len(val_data)):
    #     X_val.append(val_data[i-seq_len:i])
    #     y_val.append(val_data[:, 3][i])
    # X_val, y_val = np.array(X_val), np.array(y_val)

    # ###############################################################################

    # # Test data
    # X_test, y_test = [], []
    # for i in range(seq_len, len(test_data)):
    #     X_test.append(test_data[i-seq_len:i])
    #     y_test.append(test_data[:, 3][i])    
    # X_test, y_test = np.array(X_test), np.array(y_test)

    # print('Training set shape', X_train.shape, y_train.shape)
    # print('Validation set shape', X_val.shape, y_val.shape)
    # print('Testing set shape' ,X_test.shape, y_test.shape)

    # class Time2Vector(Layer):
    #     def __init__(self, seq_len, **kwargs):
    #         super(Time2Vector, self).__init__()
    #         self.seq_len = seq_len

    #     def build(self, input_shape):
    #         '''Initialize weights and biases with shape (batch, seq_len)'''
    #         self.weights_linear = self.add_weight(name='weight_linear',
    #                                     shape=(int(self.seq_len),),
    #                                     initializer='uniform',
    #                                     trainable=True)
            
    #         self.bias_linear = self.add_weight(name='bias_linear',
    #                                     shape=(int(self.seq_len),),
    #                                     initializer='uniform',
    #                                     trainable=True)
            
    #         self.weights_periodic = self.add_weight(name='weight_periodic',
    #                                     shape=(int(self.seq_len),),
    #                                     initializer='uniform',
    #                                     trainable=True)

    #         self.bias_periodic = self.add_weight(name='bias_periodic',
    #                                     shape=(int(self.seq_len),),
    #                                     initializer='uniform',
    #                                     trainable=True)

    #     def call(self, x):
    #         '''Calculate linear and periodic time features'''
    #         x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
    #         time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    #         time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
            
    #         time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    #         time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
    #         return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
        
    #     def get_config(self): # Needed for saving and loading model with custom layer
    #         config = super().get_config().copy()
    #         config.update({'seq_len': self.seq_len})
    #         return config

    # class SingleAttention(Layer):
    #     def __init__(self, d_k, d_v):
    #         super(SingleAttention, self).__init__()
    #         self.d_k = d_k
    #         self.d_v = d_v

    #     def build(self, input_shape):
    #         self.query = Dense(self.d_k, 
    #                         input_shape=input_shape, 
    #                         kernel_initializer='glorot_uniform', 
    #                         bias_initializer='glorot_uniform')
            
    #         self.key = Dense(self.d_k, 
    #                         input_shape=input_shape, 
    #                         kernel_initializer='glorot_uniform', 
    #                         bias_initializer='glorot_uniform')
            
    #         self.value = Dense(self.d_v, 
    #                         input_shape=input_shape, 
    #                         kernel_initializer='glorot_uniform', 
    #                         bias_initializer='glorot_uniform')

    #     def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    #         q = self.query(inputs[0])
    #         k = self.key(inputs[1])

    #         attn_weights = tf.matmul(q, k, transpose_b=True)
    #         attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
    #         attn_weights = tf.nn.softmax(attn_weights, axis=-1)
            
    #         v = self.value(inputs[2])
    #         attn_out = tf.matmul(attn_weights, v)
    #         return attn_out    

    # #############################################################################

    # class MultiAttention(Layer):
    #     def __init__(self, d_k, d_v, n_heads):
    #         super(MultiAttention, self).__init__()
    #         self.d_k = d_k
    #         self.d_v = d_v
    #         self.n_heads = n_heads
    #         self.attn_heads = list()

    #     def build(self, input_shape):
    #         for n in range(self.n_heads):
    #             self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
                
    #             # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
    #             self.linear = Dense(input_shape[0][-1], 
    #                                 input_shape=input_shape, 
    #                                 kernel_initializer='glorot_uniform', 
    #                                 bias_initializer='glorot_uniform')

    #     def call(self, inputs):
    #         attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    #         concat_attn = tf.concat(attn, axis=-1)
    #         multi_linear = self.linear(concat_attn)
    #         return multi_linear   

    # #############################################################################

    # class TransformerEncoder(Layer):
    #     def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
    #         super(TransformerEncoder, self).__init__()
    #         self.d_k = d_k
    #         self.d_v = d_v
    #         self.n_heads = n_heads
    #         self.ff_dim = ff_dim
    #         self.attn_heads = list()
    #         self.dropout_rate = dropout

    #     def build(self, input_shape):
    #         self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
    #         self.attn_dropout = Dropout(self.dropout_rate)
    #         self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    #         self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    #         # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
    #         self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
    #         self.ff_dropout = Dropout(self.dropout_rate)
    #         self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
        
    #     def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    #         attn_layer = self.attn_multi(inputs)
    #         attn_layer = self.attn_dropout(attn_layer)
    #         attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    #         ff_layer = self.ff_conv1D_1(attn_layer)
    #         ff_layer = self.ff_conv1D_2(ff_layer)
    #         ff_layer = self.ff_dropout(ff_layer)
    #         ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    #         return ff_layer 

    #     def get_config(self): # Needed for saving and loading model with custom layer
    #         config = super().get_config().copy()
    #         config.update({'d_k': self.d_k,
    #                     'd_v': self.d_v,
    #                     'n_heads': self.n_heads,
    #                     'ff_dim': self.ff_dim,
    #                     'attn_heads': self.attn_heads,
    #                     'dropout_rate': self.dropout_rate})
    #         return config          

    # def create_model():
    #     '''Initialize time and transformer layers'''
    #     time_embedding = Time2Vector(seq_len)
    #     attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    #     attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    #     attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    #     '''Construct model'''
    #     in_seq = Input(shape=(seq_len, 5))
    #     x = time_embedding(in_seq)
    #     x = Concatenate(axis=-1)([in_seq, x])
    #     x = attn_layer1((x, x, x))
    #     x = attn_layer2((x, x, x))
    #     x = attn_layer3((x, x, x))
    #     x = GlobalAveragePooling1D(data_format='channels_first')(x)
    #     x = Dropout(0.1)(x)
    #     x = Dense(64, activation='relu')(x)
    #     x = Dropout(0.1)(x)
    #     out = Dense(1, activation='linear')(x)

    #     model = Model(inputs=in_seq, outputs=out)
    #     model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    #     return model


    # model = create_model()
    # model.summary()

    # print(X_train)
    

    # callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
    #                                             monitor='val_loss', 
    #                                             save_best_only=True, verbose=1)

    # history = model.fit(X_train, y_train, 
    #                     batch_size=batch_size, 
    #                     epochs=35, 
    #                     callbacks=[callback],
    #                     validation_data=(X_val, y_val))  

    # model = tf.keras.models.load_model('Transformer+TimeEmbedding.hdf5',
    #                                custom_objects={'Time2Vector': Time2Vector, 
    #                                                'SingleAttention': SingleAttention,
    #                                                'MultiAttention': MultiAttention,
    #                                                'TransformerEncoder': TransformerEncoder})

    # train_pred = model.predict(X_train)
    # val_pred = model.predict(X_val)
    # test_pred = model.predict(X_test)

    # #Print evaluation metrics for all datasets
    # train_eval = model.evaluate(X_train, y_train, verbose=0)
    # val_eval = model.evaluate(X_val, y_val, verbose=0)
    # test_eval = model.evaluate(X_test, y_test, verbose=0)
    # print(' ')
    # print('Evaluation metrics')
    # print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
    # print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
    # print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))



    # WAVELET TRANSFORMS WORKING SECTION

    # predictor = "KNN1"

    # for symbol in load_save_symbols:
    #     df = get_proper_df(symbol, 4000, "V2")
    #     df = modify_dataframe(params["KNN1"]["FEATURE_COLUMNS"], df, True)
    # data_dict = load_all_data(symbol, params, df, get_current_datetime())

    
    # def calculate_entropy(list_values):
    #     counter_values = Counter(list_values).most_common()
    #     probabilities = [elem[1]/len(list_values) for elem in counter_values]
    #     entropy=scipy.stats.entropy(probabilities)
    #     return entropy
 
    # def calculate_statistics(list_values):
    #     n5 = np.nanpercentile(list_values, 5)
    #     n25 = np.nanpercentile(list_values, 25)
    #     n75 = np.nanpercentile(list_values, 75)
    #     n95 = np.nanpercentile(list_values, 95)
    #     median = np.nanpercentile(list_values, 50)
    #     mean = np.nanmean(list_values)
    #     std = np.nanstd(list_values)
    #     var = np.nanvar(list_values)
    #     rms = np.nanmean(np.sqrt(list_values**2))
    #     return [n5, n25, n75, n95, median, mean, std, var, rms]
    
    # def get_features(list_values):
    #     entropy = calculate_entropy(list_values)
    #     statistics = calculate_statistics(list_values)
    #     # print(entropy)
    #     # print(statistics)
    #     return [entropy] + statistics

    # def load_ecg_data(filename):
    #     raw_data = loadmat(filename)
    #     list_signals = raw_data['ECGData'][0][0][0]
    #     list_labels = list(map(lambda x: x[0][0], raw_data['ECGData'][0][0][1]))
    #     return list_signals, list_labels
    
    

    # def get_ecg_features(ecg_data, waveletname):
    #     list_features = []
    #     # print(list_labels)
    #     i = 0
    #     for signal in ecg_data: # 97 
    #         # print(f"\n\n SIGNAL {signal} {signal.shape} {i} SIGNAL \n")
    #         list_coeff = pywt.wavedec(signal, waveletname)
    #         features = []
    #         for coeff in list_coeff:
    #             # print(len(coeff), coeff.shape)
    #             features += get_features(coeff)
    #             # print(len(get_features(coeff)))
    #         # print(len(features))
    #         list_features.append(features)
    #         i += 1

    #     list_features = sum(list_features, [])
    #     return list_features


    # TRYING 2D data with ECG FEATURES
    # fut = df["c"].shift(-1)
    # fut = list(map(lambda current, future: int(float(future) > float(current)), df['c'][:-1], df['c'][1:]))
    # print(df.tail(10))
    # df = df.shift(1).dropna()
    # df["fut"] = fut
    # print(df.tail(20))
    
    # data = df.to_numpy()
    # print(f" OG data shape {data.shape}")


    # print(df.head(100))
    # print(len(fut), len(df['c']))
    # print(fut, type(fut))

    # x, y = get_ecg_features(data, fut, "db4")
    # x = np.array(x)
    # y = np.array(y)
    # print(x.shape, y.shape)


    # TRYING 3D DATA WITH ECG_FEATURES
    # future = list(map(lambda current, future: int(float(future) > float(current)), df['c'][:-1], df['c'][1:]))
    # df["future"] = df['c'].shift(-1)
    # df = df.dropna()
    # df["future"] = future
    # print(df)

    # if "c" not in params['KNN1']["FEATURE_COLUMNS"]:
    #     df = df.drop(columns=["c"])

    # result = {}

    # sequence_data = []
    # sequences = deque(maxlen=100)
    # for entry, target in zip(df[params['KNN1']["FEATURE_COLUMNS"]].values, df["future"].values):
    #     sequences.append(entry)
    #     if len(sequences) == 100:
    #         sequence_data.append([np.array(sequences), target])
    
    # # construct the X"s and y"s
    # X, y = [], []
    
    # for seq, target in sequence_data:
    #     X.append(seq)
    #     y.append(target)

    # X = np.array(X)
    # y = np.array(y)

    # X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # print(X)
    # print(X.shape)
    # print(y)

    # for day in X:
    #     features = get_ecg_features(day, "db4")
    #     features = np.array(features)
    #     print(features.shape)

    # CHECK NEWS RESULTS
    
    # base_df = get_proper_df("AGYS", 4000, "V2")
    # news_df = modify_dataframe("SPY", params2["nn1"]["FEATURE_COLUMNS"], base_df, params2["nn1"]["TEST_VAR"], "V2", False)

    # print(news_df.tail(60))

    # top_10 = []

    # top_20 = []

    # top_50 = []

    # top_100 = []

    # top_300 = []
    
    # top_500 = []

    # news = api.get_news(symbol="SPY", limit=1000)
    # print(news)
    # print(f"{len(news)} news articles were found for")


    # GET ACCOUNT ACTIVITIES

    # api = get_api()

    # port_history = api.get_portfolio_history()
    # get_act = api.get_activities(page_size=100)

    # print(len(port_history))
    # print(len(get_act))
    # print(port_history[-1])
    # print(get_act)
    # print(type(get_act[-1]))
    

    # help = port_history.df
    # print(help)
    

    # import json
    # f = open("alpaca_sucks_porfolio_history.json", "w")
    # f.write(str(port_history))
    # f.close()

    # TIINGO SHIT

    symbol = "UPS"
    # historical_date_str = "2015-12-1"
    # historical_date_str = ""
    latest_date_str = get_current_date_string()

    # for symbol in load_save_symbols:

    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?endDate={latest_date_str}&token={tiingo_key}"
    print(url)

    r = requests.get(url)
    # print(r.json())
    response = r.json()
    response = pd.DataFrame(response)
    response = response.set_index(["date"])
    response.index = pd.to_datetime(response.index).date

    
    response = response.drop(columns=["adjHigh", "adjLow", "adjOpen", "adjVolume", "splitFactor", "divCash"], axis=1) 
    # response = response.rename(columns = {"adjHigh":"high","adjLow":"low","adjOpen":"open","adjVolume":"volume"}) #"adjClose":"close"
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    print(response)

    #     alpaca_df = get_proper_df(symbol, 4000, "V2")

    #     print(type(alpaca_df.index[0]), print(type(response.index[0])))

    #     printing_df = pd.concat([alpaca_df['c'], response['close'], response["adjClose"]], axis=1, keys=["alpaca_c", "tiingo_c", "tiingo_adj_c"])
        
    #     # print(response["close"])
    #     # print(printing_df)

    #     print(len(alpaca_df["c"]), len(response["close"]), len(response["adjClose"]))
    #     print(get_percent_away(alpaca_df['c'], response['close']))
    #     print(get_percent_away(alpaca_df['c'], response['adjClose']))

    ##########################################################
    # Rework sav_gol so it doesn't cheat

    # df = get_proper_df("AGYS", 4000, "V2")
    # print(params2["nn1"]["FEATURE_COLUMNS"])
    # df = modify_dataframe("AGYS", params2["nn1"]["FEATURE_COLUMNS"], df, params2["nn1"]["TEST_VAR"], "V2", True)

    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # # print(df)

    # plt.plot(df["c"], c="b")
    # plt.plot(df["es.c"], c="r")
    # # plt.plot(df["testing"], c="g")
    # plt.show()

    print(time.perf_counter() - s, flush=True)

    #best so far for volitility 
    # symbol = "UVXY"
    # Dow Jones = "DIA"
    # S&P 500 = "SPY"
    # NASDAQ = "QQQ"
    
    # print(get_proper_df(symbol, 4000, "V2"))

    all_features = ["o", "l", "h", "c", "m", "v", "up_band", "low_band", "OBV", "RSI", "lin_reg", "lin_reg_ang", 
        "lin_reg_int", "lin_reg_slope", "pears_cor", "mon_flow_ind", "willR", "std_dev", "min", "max", "commod_chan_ind", "para_SAR", "para_SAR_ext", "rate_of_change", 
        "ht_dcperiod", "ht_trendmode", "ht_dcphase", "ht_inphase", "quadrature", "ht_sine", "ht_leadsine", "ht_trendline", "mom", "abs_price_osc", "KAMA", "typ_price", 
        "ult_osc", "chai_line", "chai_osc", "norm_avg_true_range", "median_price", "var", "aroon_down", "aroon_up", "aroon_osc", "bal_of_pow", "chande_mom_osc", "MACD", 
        "MACD_signal", "MACD_hist", "con_MACD", "con_MACD_signal", "con_MACD_hist", "fix_MACD", "fix_MACD_signal", "fix_MACD_hist", "min_dir_ind", "min_dir_mov", "plus_dir_ind",
        "plus_dir_mov", "per_price_osc", "stoch_fast_k", "stoch_fast_d", "stoch_rel_stren_k", "stoch_rel_stren_d", "stoch_slowk", "stoch_slowd", "TRIX",
        "weigh_mov_avg", "DEMA", "EMA", "MESA_mama", "MESA_fama", "midpnt", "midprice", "triple_EMA", "tri_MA", "avg_dir_mov_ind", "true_range", "avg_price",
        "weig_c_price", "beta", "TSF", "day_of_week", "vad", "fin_vad", "fin_bert_pos", "garman_klass", "hodges_tompkins", "kurtosis", "parkison", "rogers_stachell", "skew",
        "yang_zhang"]

    direct_value_features = ["o", "l", "h", "c", "m", "up_band", "low_band", "lin_reg", "lin_reg_ang", "lin_reg_int", "lin_reg_slope", 
                "min", "max", "ht_trendline",  "KAMA", "typ_price", "median_price", "var", "TRIX", "weigh_mov_avg", "DEMA", "EMA", "MESA_mama", "MESA_fama", 
                "midpnt", "midprice", "triple_EMA", "tri_MA", "avg_price", "weig_c_price", "TSF"]

    

    def store_sci_predictors(params, data_dict, saved_models):
        for predictor in params["ENSEMBLE"]:
            if "DTREE" in predictor:
                tree = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
                    min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"])
                tree.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])     
                saved_models[predictor] = tree         
            # elif "RFORE" in predictor:

            # elif "KNN" in predictor:

        return saved_models


    # year = 2019
    # month = 12
    # day = 1
    # how_damn_long_to_run_for = 500
 
    # year = 2020
    # month = 5
    # day = 17
    # how_damn_long_to_run_for = 250

    # tuning(year, month, day, how_damn_long_to_run_for, params)

    # backtest_comparator(5, 9, "sav_gol", 3000)
    # fuck_me_symbols = ["AGYS", "AMKR","BG", "BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
    #     "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]
    # for symbol in fuck_me_symbols:
    #     df, blah, bal, alalal = load_3D_data(symbol, params["nn1"], to_print=False)
    #     print(symbol)
    #     print(f"{symbol}: {pre_c_comparator(df, 3000)}", flush=True)


    

