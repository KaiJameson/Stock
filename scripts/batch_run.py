from backtest import back_testing
from tuner import tuning
from functions.functions import check_directories
from config.symbols import test_year, test_month, test_day, test_days, btest_list, tune_list
import sys

check_directories()


if len(sys.argv) > 1:
    if "btest" in sys.argv[1]:
        for dictionary in btest_list:
            if dictionary["SAVE_FOLDER"] == sys.argv[1]:
                back_testing(test_year, test_month, test_day, test_days, dictionary)
    elif "tune" in sys.argv[1]:
        for dictionary in tune_list:
            if dictionary["SAVE_FOLDER"] == sys.argv[1]:
                tuning(test_year, test_month, test_day, test_days, dictionary)
    else:
        print("You must give this program an argument in the style of \"btest#\"")
        print("or \"tune#\"So that it knows what folder to save your models into.")
        print("Please try again")
        sys.exit(-1)

else:
    print("You need to provide a second argument for this file. This argument is either")
    print("btest# or tune#")
    sys.exit(-1)


    


