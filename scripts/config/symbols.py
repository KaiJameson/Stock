import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, GRU

decision_symbols = ["MSFT"]

load_save_symbols = ["AGYS", "AMKR", "BG","BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
"INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]


# load_save_symbols = ["AXL", "FOLD", "DISCA", "EGHT", "UNFI", "KTOS", "GT", "BWA", "ZIXI", "ACIW",
# "PENN", "PRTS", "IIVI", "LUV", "WEN", "TSN", "AMD", "SJM", "HD", "FCX", "SCSC", "FORM", "TXT", "HUN",
# "SBUX", "UPS", "AES", "XRX", "WCC", "AGYS", "AMKR", "BG","BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
# "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]

real_test_symbols = ["AGYS", "AMKR", "BG", "BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
"INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]

# real_test_symbols = ["AGYS"]

# test day for all the stocks over 30 days
test_year = 2021
test_month = 6
test_day = 27

# test day for telling a longer term look at fewer stocks
# test_year = 2020
# test_month = 6
# test_day = 4

# test day for predicting a stocks performance over the majority of 2020
# test_year = 2020
# test_month = 2
# test_day = 10

test_days = 30


batch_run_list = [

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.5,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch1"
    },    

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.55,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch1"
    },

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.6,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch1"
    },

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.65,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch2"
    },

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.7,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch2"
    },

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.75,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch2"
    },

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.8,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch3"
    },

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.85,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch3"
    },

    {
    "N_STEPS": 100,
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": 256,
    "DROPOUT": 0.9,
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": 2000,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "batch3"
    }
    ]

tune_sym_dict = {
    "tuning1": ["AGYS", "AMKR", "BG", "BGS", "CAKE"],
    "tuning2": ["CCJ", "DFS", "ELY", "FLEX", "INTC"],
    "tuning3": ["JBLU", "LLNW", "NWL", "QCOM", "RDN"],
    "tuning4": [],
    "tuning5": []

}


tune_year = 2020
tune_month = 5
tune_day = 17

tune_days = 250

do_the_trades = True
trading_real_money = True








