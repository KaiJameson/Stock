from backtest import back_testing
from tuner import tuning
from functions.functions import check_directories
from config.symbols import test_year, test_month, test_day, test_days, base_dict, modifications
import sys

check_directories()


if len(sys.argv) > 2:
    for test in modifications:
        tuning(test_year, test_month, test_day, test_days, dictionary)
else:
    print("You must give this program two arguments in the style of \"tune#\"")
    print("then \"sym#\" So that it knows tests to run and what symbols to use.")
    print("Please try again")
    sys.exit(-1)



    


