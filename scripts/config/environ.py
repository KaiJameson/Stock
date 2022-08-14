from config.silen_ten import silence_tensorflow
silence_tensorflow()

from tensorflow.keras.layers import LSTM, Dense

directory_dict = {
    "backtest":          "../backtest",
    "batch_run":         "../batch_run",
    "batch_run_summary": "../batch_run/summary",
    "comparator_results":"../tuning_info/comparators",
    "config":            "../config",
    "data":              "../data", 
    "day_summary":       "../excel/day_summary",
    "excel":             "../excel",
    "graph":             "../plots",
    "keras_tuner":       "../keras_tuner",
    "load_run_results":  "../excel/load_run_results",
    "model":             "../models",
    "runtime_predict":   "../excel/runtime_predict",
    "PL":                "../excel/profit_loss",
    "reports":           "../reports",
    "results":           "results",
    "runtime_price":     "../excel/curr_price",
    "tax":               "../tax",
    "trade_perform":     "../excel/trade_perform",
    "tune_summary":      "../tuning_info/summary",
    "tuning":            "../tuning_info",
    "save_predicts":     "../predictions",
    "sim_trades":        "../sim_trades",
}


error_file = "../error_file.txt"

time_zone = "US/EASTERN"
test_money = 10000.0
stocks_traded = 20
random_seed = 314
back_test_days = 100
save_logs = False
to_plot = False
make_config = False


"""
# N_STEPS = Window size or the sequence length.
# Lookup step = How many days the model will be trying to predict
# into the future.
# TEST_SIZE = How much of the data will be split between validation
# and testing. 0.2 is 20%.
# LAYERS = The Architechure of the model
# CELL = Type of cell used in each layer.
# DROPOUT = % dropout, cells that are dropped in that training batch
# will not be used to make the output. 
# BIDIRECTIONAL = Whether or not the LSTM cells" memory can flow 
# forward in time if False or forwards and backwards in time
# LOSS = "huber_loss"
# OPTIMIZER = "adam"
# BATCH_SIZE = How many sets of data are ran together.
# EPOCHS = How many times the machine trains.
# PATIENCE = How many epochs of no improvement in the validation 
# loss it takes before the training loop is ended early.
# SAVELOAD = Whether or not to temporarily save the weights of the 
# model or to save the whole thing 
# LIMIT = how many days of data do you want 
# FEATURE_COLUMNS = what types of data do you want to include for
# your model to train on
"""

defaults = {
    "ENSEMBLE": ["nn8", "nn11", "nn14"],
    "TRADING": True,
    "SAVE_FOLDER": "trading",
    "nn8" : { 
        "N_STEPS": 100,
        "LOOKUP_STEP": 1,
        "TEST_SIZE": 0.2,
        "LAYERS": [(256, LSTM), (256, Dense), (128, Dense), (64, Dense)],
        "DROPOUT": .4,
        "BIDIRECTIONAL": False,
        "LOSS": "huber_loss",
        "OPTIMIZER": "adam",
        "BATCH_SIZE": 1024,
        "EPOCHS": 2000,
        "PATIENCE": 200,
        "LIMIT": 4000,
        "FEATURE_COLUMNS": ["s.o", "s.l", "s.h", "s.c", "s.m", "s.v", "tc", "vwap"],
        "SHUFFLE": True,
        "TEST_VAR": "c",
        "SAVE_PRED": {}
        },
    "nn11" : { 
        "N_STEPS": 100,
        "LOOKUP_STEP": 1,
        "TEST_SIZE": 0.2,
        "LAYERS": [(256, LSTM), (256, Dense), (128, Dense), (64, Dense)],
        "DROPOUT": .4,
        "BIDIRECTIONAL": False,
        "LOSS": "huber_loss",
        "OPTIMIZER": "adam",
        "BATCH_SIZE": 1024,
        "EPOCHS": 2000,
        "PATIENCE": 200,
        "LIMIT": 4000,
        "FEATURE_COLUMNS": ["pc.o", "pc.l", "pc.h", "pc.c", "pc.m", "pc.v", "tc", "vwap"],
        "SHUFFLE": True,
        "TEST_VAR": "c",
        "SAVE_PRED": {}
        },
    "nn14" : { 
        "N_STEPS": 100,
        "LOOKUP_STEP": 1,
        "TEST_SIZE": 0.2,
        "LAYERS": [(256, LSTM), (256, Dense), (128, Dense), (64, Dense)],
        "DROPOUT": .4,
        "BIDIRECTIONAL": False,
        "LOSS": "huber_loss",
        "OPTIMIZER": "adam",
        "BATCH_SIZE": 1024,
        "EPOCHS": 2000,
        "PATIENCE": 200,
        "LIMIT": 4000,
        "FEATURE_COLUMNS": ["s.o", "s.l", "s.h", "s.c", "s.m", "s.v", "pc.o", "pc.l", "pc.h", "pc.c", "pc.m", "pc.v", "tc", "vwap"],
        "SHUFFLE": True,
        "TEST_VAR": "c",
        "SAVE_PRED": {}
        }, 
    "LIMIT": 4000
}

