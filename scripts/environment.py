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

defaults = {
    'N_STEPS': 300,
    'LOOKUP_STEP': 1,
    'TEST_SIZE': 0.2,
    'N_LAYERS': 3,
    'CELL': LSTM,
    'UNITS': 448,
    'DROPOUT': 0.3,
    'BIDIRECTIONAL': True,
    'LOSS': 'huber_loss',
    'OPTIMIZER': 'adam',
    'BATCH_SIZE': 64,
    'EPOCHS': 2000,
}




