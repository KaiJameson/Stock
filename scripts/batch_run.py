from backtest import back_testing
from environ import error_file
from functions import check_directories
from symbols import test_year, test_month, test_day, test_days, batch_run_list
import time
import sys

check_directories()


if len(sys.argv) > 1:
    if sys.argv[1] == "batch1":
        pass
    elif sys.argv[1] == "batch2":
        pass
    elif sys.argv[1] == "batch3":
        pass
    elif sys.argv[1] == "batch4":
        pass
    elif sys.argv[1] == "batch5":
        pass
    else:
        print("You must give this program an argument in the style of \"batcht#\"")
        print("So that it knows what folder to save your models into.")
        print("Please try again")
        sys.exit(-1)


else:
    print("You need to provide a second argument that says which batch ")
    print("you want to use. Please try again")
    sys.exit(-1)

for dictionary in batch_run_list:
    print(dictionary["SAVE_FOLDER"])
    if dictionary["SAVE_FOLDER"] == sys.argv[1]:
        back_testing(test_year, test_month, test_day, test_days, dictionary)
    


