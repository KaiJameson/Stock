from alpaca_nn_functions import load_data, predict, getOwnedStocks, decide_trades, return_real_predict, get_all_accuracies, nn_report, make_excel_file, get_all_maes
from symbols import load_save_symbols, do_the_trades
from environment import model_saveload_directory, error_file, config_directory, defaults, test_var
from functions import check_directories
from tensorflow.keras.models import load_model
import traceback
import time
import os
import sys

check_directories()

def load_trade(symbols):
    
    owned = getOwnedStocks()
    for symbol in symbols:
        try:
            start_time = time.time()
            config_name = config_directory + "/" + symbol + ".csv"
            if os.path.isfile(config_name):
                f = open(config_name, "r")
                values = {}
                for line in f:
                    parts = line.strip().split(",")
                    values[parts[0]] = parts[1]
                N_STEPS = int(values["N_STEPS"])
                data, train, valid, test = load_data(symbol, int(values["N_STEPS"]), shuffle=False)
            else:
                N_STEPS = int(defaults["N_STEPS"])
                data, train, valid, test = load_data(symbol, int(defaults["N_STEPS"]), shuffle=False)

            LOOKUP_STEP = defaults["LOOKUP_STEP"]

            model = load_model(model_saveload_directory + "/" + symbol + ".h5")
            
            train_acc, valid_acc, test_acc = get_all_accuracies(model, data, LOOKUP_STEP)
            
            test_mae, valid_mae, train_mae = get_all_maes(model, test, valid, train, data) 

            total_time = time.time() - start_time
            percent = nn_report(symbol, total_time, model, data, test_acc, valid_acc, train_acc, test_mae, valid_mae, train_mae, N_STEPS)

            

            if do_the_trades:
                decide_trades(symbol, owned, test_acc, percent)
            else:
                print("Why are you running this if you don't want to do the trades?")

            print("Finished running: " + symbol)

        except KeyboardInterrupt:
            print("I acknowledge that you want this to stop")
            print("Thy will be done")
            sys.exit(-1)
        except:
            f = open(error_file, "a")
            f.write("problem with configged stock: " + symbol + "\n")
            exit_info = sys.exc_info()
            f.write(str(exit_info[1]) + "\n")
            traceback.print_tb(tb=exit_info[2], file=f)
            f.close()
            print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

s = time.time()
load_trade(load_save_symbols)
make_excel_file()
tt = (time.time() - s) / 60
print("In total it took " + str(round(tt, 2)) + " minutes to run all the files.")
