from api_key import real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key
from alpaca_nn_functions import (load_data, predict, getOwnedStocks, return_real_predict, 
get_all_accuracies, nn_report,  percent_from_real, buy_all_at_once, create_model)
from symbols import load_save_symbols, do_the_trades
from environment import model_saveload_directory, error_file, config_directory, defaults, test_var
from functions import check_directories, make_excel_file, make_load_run_excel
from error_functs import error_handler
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model
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

            model_name = (symbol + "-" + str(defaults["FEATURE_COLUMNS"]) + "-limit-" + str(defaults["LIMIT"]) +"-n_step-" + str(defaults["N_STEPS"]) 
            + "-layers-" + str(defaults["N_LAYERS"]) + "-units-" + str(defaults["UNITS"]) + "-epochs-" + str(defaults["EPOCHS"]))

            print("\n~~~Now Starting " + symbol + "~~~")
            
            time_s = time.time()
            data, train, valid, test = load_data(symbol, n_steps=defaults["N_STEPS"], batch_size=defaults["BATCH_SIZE"],
            limit=defaults["LIMIT"], feature_columns=defaults["FEATURE_COLUMNS"], shuffle=False, to_print=False)
            print("Loading the data took " + str(time.time() - time_s) + " seconds")    

            time_s = time.time()
            model = create_model(defaults["N_STEPS"], defaults["UNITS"], defaults["CELL"], defaults["N_LAYERS"], 
                defaults["DROPOUT"],  defaults["LOSS"], defaults["OPTIMIZER"], defaults["BIDIRECTIONAL"]
            )
            model.load_weights(model_saveload_directory + "/" + defaults["SAVE_FOLDER"] + "/" + model_name + ".h5")
            # model.summary()
            print("Loading the model took " + str(time.time() - time_s) + " seconds")    

            time_s = time.time()
            train_acc, valid_acc, test_acc = get_all_accuracies(model, data, defaults["LOOKUP_STEP"])
            print("Getting the accuracies took " + str(time.time() - time_s) + " seconds")   

            total_time = time.time() - start_time
            time_s = time.time()
            percent, future_price = nn_report(symbol, total_time, model, data, test_acc, valid_acc, train_acc, defaults["N_STEPS"])
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
        except Exception:
            error_handler(symbol, Exception)

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
