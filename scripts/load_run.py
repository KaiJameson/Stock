from api_key import real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key
from alpaca_nn_functions import (load_data, predict, getOwnedStocks, decide_trades, return_real_predict, 
get_all_accuracies, nn_report,  percent_from_real, buy_all_at_once, create_model)
from symbols import load_save_symbols, do_the_trades
from environment import model_saveload_directory, error_file, config_directory, defaults, test_var
from functions import check_directories, make_excel_file, make_load_run_excel
from tensorflow.keras.models import load_model
import traceback
import time
import os
import sys

check_directories()

def load_trade(symbols):
    
    owned = getOwnedStocks()

    price_prediction_list = {}
    for symbol in symbols:
        try:
            start_time = time.time()
            print("\n~~~Now Starting " + symbol + "~~~")

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
                time_s = time.time()
                data, train, valid, test = load_data(symbol, int(defaults["N_STEPS"]), shuffle=False)
                print("Loading the data took " + str(time.time() - time_s) + " seconds")    

            LOOKUP_STEP = defaults["LOOKUP_STEP"]

            time_s = time.time()
            model = create_model(N_STEPS, loss=defaults["LOSS"], units=defaults["UNITS"], 
            cell=defaults["CELL"], n_layers=defaults["N_LAYERS"], dropout=defaults["DROPOUT"], 
            optimizer=defaults["OPTIMIZER"], bidirectional=defaults["BIDIRECTIONAL"])
            model.load_weights(model_saveload_directory + "/" + symbol + ".h5")
            # model.summary()
            print("Loading the model took " + str(time.time() - time_s) + " seconds")    

            time_s = time.time()
            train_acc, valid_acc, test_acc = get_all_accuracies(model, data, LOOKUP_STEP)
            print("Getting the accuracies took " + str(time.time() - time_s) + " seconds")   

            total_time = time.time() - start_time
            time_s = time.time()
            percent, future_price = nn_report(symbol, total_time, model, data, test_acc, valid_acc, train_acc, N_STEPS)
            price_prediction_list[symbol] = future_price
            print("NN report took " + str(time.time() - time_s) + " seconds")

            y_real, y_pred = return_real_predict(model, data["X_valid"], data["y_valid"], data["column_scaler"][test_var])
            make_load_run_excel(symbol, train_acc, valid_acc, test_acc, percent_from_real(y_real, y_pred), abs((percent - 1) * 100))
           
            print("Finished running: " + symbol)

            sys.stdout.flush()

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

    if do_the_trades:
        time_s = time.time()
        buy_all_at_once(symbols, owned, price_prediction_list)
        print("Performing all the trades took " + str(time.time() - time_s) + " seconds")
    else:
        print("Why are you running this if you don't want to do the trades?")

s = time.time()
load_trade(load_save_symbols)
time_s = time.time()
make_excel_file()
print("Making the excel file took " + str(time.time() - time_s) + " seconds\n")
tt = (time.time() - s) / 60
print("In total it took " + str(round(tt, 2)) + " minutes to run all the files.")
