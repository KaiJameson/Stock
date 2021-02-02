from real_test import the_real_test
from environment import error_file
from functions import check_directories
from symbols import test_year, test_month, test_day, test_days
from tensorflow.keras.layers import LSTM
import time
import sys

check_directories()

the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume"],
n_steps=100, lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=100, dropout=.3, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=2000)

the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "7_moving_avg"],
n_steps=100, lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=100, dropout=.3, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=2000)

the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "day_of_week"],
n_steps=100, lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=100, dropout=.3, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=2000)

# the_real_test(test_year, test_month, test_day, test_days, 
# ["open", "low", "high", "close", "mid", "volume", "midprice"],
# n_steps=300, lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
# bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
# epochs=800, patience=200, saveload=True, limit=4000)

# the_real_test(test_year, test_month, test_day, test_days, 
# ["open", "low", "high", "close", "mid", "volume", "pearsons_correl"],
# n_steps=300, lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
# bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
# epochs=800, patience=200, saveload=True, limit=4000)


# the_real_test(test_year, test_month, test_day, test_days, 
# ["open", "low", "high", "close", "mid", "volume", "lin_regres"],
# n_steps=300, lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
# bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
# epochs=800, patience=200, saveload=True, limit=4000)





