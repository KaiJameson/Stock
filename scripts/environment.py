from tensorflow.keras.layers import LSTM

test_money = 10000
test_var = 'close'
reports_directory = '../reports'
error_file = '../error_file.txt'
config_directory = '../config'
stock_decisions_directory = reports_directory + '/decisions'
graph_directory = '../plots'
random_seed = 314
trades_dir = '../trades'
tuning_directory = '../tuning_info'
back_test_days = 100
data_directory = '../data'
save_logs = False
do_the_trades = False

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

defaults = {
    'N_STEPS': 300,
    'LOOKUP_STEP': 1,
    'TEST_SIZE': 0.2,
    'N_LAYERS': 3,
    'CELL': LSTM,
    'UNITS': 448,
    'DROPOUT': 0.3,
    'BIDIRECTIONAL': False,
    'LOSS': 'huber_loss',
    'OPTIMIZER': 'adam',
    'BATCH_SIZE': 64,
    'EPOCHS': 2000,
    'PATIENCE': 500,
}




