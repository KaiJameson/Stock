from real_test import the_real_test
from environment import error_file
from functions import check_directories
from symbols import test_year, test_month, test_day, test_days
import time
import sys

check_directories()

time_s = time.time()
print("test 1")
the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "7_moving_avg"],
lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=4000)
print("test 1 took " + str(time.time() - time_s))

time_s = time.time()
print("test 2")
the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "lin_regres"],
lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=4000)
print("test 2 took " + str(time.time() - time_s))

time_s = time.time()
print("test 3")
the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "lin_regres"],
lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=4000)
print("test 3 took " + str(time.time() - time_s))

time_s = time.time()
print("test 4")
the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "lin_regres"],
lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=4000)
print("test 4 took " + str(time.time() - time_s))

time_s = time.time()
print("test 5")
the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "lin_regres"],
lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=4000)
print("test 5 took " + str(time.time() - time_s))

time_s = time.time()
print("test 6")
the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "lin_regres"],
lookup_step=1, test_size=.2, n_layers=2, cell=LSTM, units=256, dropout=.4, 
bidirectional=False, loss="huber_loss", optimizer="adam", batch_size=256, 
epochs=800, patience=200, saveload=True, limit=4000)
print("test 6 took " + str(time.time() - time_s))




