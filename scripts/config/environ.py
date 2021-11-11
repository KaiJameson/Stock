from config.silen_ten import silence_tensorflow
silence_tensorflow()

from tensorflow.keras.layers import LSTM

directory_dict = {
    "backtest_dir":        "../backtest",
    "config_dir":          "../config",
    "excel_dir":           "../excel",
    "graph_dir":           "../plots",
    "load_run_results":    "../excel/load_run_results",
    "model_dir":           "../models",
    "runtime_predict_dir":  "../excel/runtime_predict",
    "PL_dir":              "../excel/profit_loss",
    "reports_dir":         "../reports",
    "results_dir":         "results",
    "runtime_price_dir":   "../excel/curr_price",
    "tax_dir":             "../tax",
    "trade_perform_dir":   "../excel/trade_perform",
    "tuning_dir":          "../tuning_info"
}

load_params = {
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "LIMIT": 200,
    "N_STEPS": 100,
    "BATCH_SIZE": 1024,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LOSS": "huber_loss"
}

error_file = "../error_file.txt"

test_var = "close"
time_zone = "US/EASTERN"
test_money = 10000.0
stocks_traded = 20
random_seed = 314
back_test_days = 100
save_logs = False
do_the_trades = True
to_plot = False
make_config = False
using_all_accuracies = False


"""
# N_STEPS = Window size or the sequence length.
# Lookup step = How many days the model will be trying to predict
# into the future.
# TEST_SIZE = How much of the data will be split between validation
# and testing. 0.2 is 20%.
# N_LAYERS = How many hidden neural layers the model will have.
# CELL = Type of cell used in each layer.
# UNITS = Number of neurons per layer.
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
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.4,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "trading"
}

