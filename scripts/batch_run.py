from backtest import back_testing
from environ import error_file
from functions import check_directories
from symbols import test_year, test_month, test_day, test_days, batch_run_list
import time
import sys

check_directories()

for dictionary in batch_run_list:
    back_testing(test_year, test_month, test_day, test_days, dictionary)


