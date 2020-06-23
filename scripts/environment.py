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
to_plot = True

'''
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
# BIDIRECTIONAL = Whether or not the LSTM cells' memory can flow 
# forward in time if False or forwards and backwards in time
# LOSS = "huber_loss"
# OPTIMIZER = "adam"
# BATCH_SIZE = How many sets of data are ran together.
# EPOCHS = How many times the machine trains.
# PATIENCE = How many epochs of no improvement in the validation 
# loss it takes before the training loop is ended early.
'''

defaults = {
    'N_STEPS': 300,
    'LOOKUP_STEP': 1,
    'TEST_SIZE': 0.2,
    'N_LAYERS': 3,
    'CELL': LSTM,
    'UNITS': 448,
    'DROPOUT': 0.4,
    'BIDIRECTIONAL': False,
    'LOSS': 'huber_loss',
    'OPTIMIZER': 'adam',
    'BATCH_SIZE': 128,
    'EPOCHS': 2000,
    'PATIENCE': 400,
}




