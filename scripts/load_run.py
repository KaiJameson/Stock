from alpaca_nn_fuctions import load_data, predict, getOwnedStocks, decide_trades
from symbols import load_save_symbols, do_the_trades
from environment import model_saveload_directory, error_file, config_directory, defaults
from tensorflow.keras.models import load_model
import os
import sys

check_directories()

def load_trade(symbols):
    owned = getOwnedStocks()
    for symbol in symbols:
        try:
            config_name = config_directory + "/" + symbol + ".csv"
            if os.path.isfile(config_name):
                f = open(config_name, "r")
                values = {}
                for line in f:
                    parts = line.strip().split(",")
                    values[parts[0]] = parts[1]
                N_STEPS = int(values["N_STEPS"])
                data, train, valid, test = load_data(symbol, int(values["N_STEPS"]), shuffle=False):
            else:
                N_STEPS = int(defaults["N_STEPS"])
                data, train, valid, test = load_data(symbol, int(defaults["N_STEPS"]), shuffle=False):

            model = load_model(model_saveload_directory + "/" + symbol + ".h5")
            
            predicted_price = predict(model, data, N_STEPS)

            y_test_real, y_test_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var])
            test_acc = get_accuracy(y_test_real, y_test_pred, LOOKUP_STEP)
            curr_price = y_test_real[-1]
            percent = predicted_price / curr_price

            if do_the_trades:
                decide_trades(symbol, owned, test_acc, percent)
            else:
                print("Why are you running this if you don't want to do the trades?")

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



load_trade(load_save_symbols)
