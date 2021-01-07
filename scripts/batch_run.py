
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
["open", "low", "high", "close", "mid", "volume", "avg_price"])
print("test 1 took " + str(time.time() - time_s))

time_s = time.time()
print("test 2")
the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "7_moving_avg"])
print("test 1 took " + str(time.time() - time_s))

time_s = time.time()
print("test 3")
the_real_test(test_year, test_month, test_day, test_days, 
["open", "low", "high", "close", "mid", "volume", "KAMA"])
print("test 1 took " + str(time.time() - time_s))




