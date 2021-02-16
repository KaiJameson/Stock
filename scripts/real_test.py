import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.layers import LSTM
from time_functions import get_short_end_date, get_year_month_day, increment_calendar
from functions import (check_directories, real_test_excel, real_test_directory, delete_files, 
interwebz_pls, delete_files_in_folder, get_test_name, read_saved_contents)
from symbols import real_test_symbols, test_year, test_month, test_day, test_days
from error_functs import error_handler
from alpaca_nn_functions import (get_api, create_model, get_all_accuracies, predict, load_data, 
load_model_with_data, return_real_predict)
from alpaca_neural_net import saveload_neural_net
from environment import error_file, model_saveload_directory, test_var, back_test_days
from statistics import mean
import pandas as pd
import datetime
import sys
import time
import ast


def the_real_test(test_year, test_month, test_day, test_days, params):

    symbol = real_test_symbols[0]
    api = get_api()
    
    test_name = get_test_name(params)
    days_done = 1
    total_days = test_days
    time_so_far = 0.0
    percent_away_list = []
    correct_direction_list = []
    epochs_list = []
    print(test_name)

    if os.path.isfile(real_test_directory + "/" + test_name + ".txt"):
        print("A fully completed file with the name " + test_name + " already exists.")
        print("Exiting the_real_test now: ")
        return

    # check if we already have a save file, if we do, extract the info and run it
    if os.path.isfile(real_test_directory + "/" + "SAVE-" + test_name + ".txt"):
        total_days, days_done, test_days, time_so_far, test_year, test_month, test_day, percent_away_list, correct_direction_list, epochs_list = read_saved_contents(real_test_directory, test_name)

        
    current_date = get_short_end_date(test_year, test_month, test_day)

    while test_days > 0:
        try:
            time_s = time.time()
            current_date = increment_calendar(current_date, api, symbol)

            for symbol in real_test_symbols:
                print("\nCurrently on day " + str(days_done) + " of " + str(total_days) + " using folder: " + params["SAVE_FOLDER"] + ".\n")
                epochs_run = saveload_neural_net(symbol, current_date, params)
                epochs_list.append(epochs_run)
                
            print("Model result progress[", end='')
            for symbol in real_test_symbols:
                # get model name for future reference
                model_name = (symbol + "-" + get_test_name(params))

                # setup to allow the rest of the values to be calculated
                data, model = load_model_with_data(symbol, current_date, params, model_saveload_directory, model_name)

                # first grab the current price by getting the latest value from the og data frame
                y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var]) 
                real_y_values = y_real[-back_test_days:]
                current_price = real_y_values[-1]

                # then use predict fuction to get predicted price
                predicted_price = predict(model, data, params["N_STEPS"])

                # get the actual price for the next day the model tried to predict by incrementing the calendar by one day
                interwebz_pls(symbol, current_date, "calendar")
                cal = api.get_calendar(start=current_date + datetime.timedelta(1), end=current_date + datetime.timedelta(1))[0]
                one_day_in_future = pd.Timestamp.to_pydatetime(cal.date).date()
                df = api.polygon.historic_agg_v2(symbol, 1, "day", _from=one_day_in_future, to=one_day_in_future).df
                actual_price = df.iloc[0]["close"]

                # get the percent difference between prediction and actual
                p_diff = round((abs(actual_price - predicted_price) / actual_price) * 100, 2)

                if ((predicted_price > current_price and actual_price > current_price) or 
                (predicted_price < current_price and actual_price < current_price)): 
                    correct_dir = 1.0
                elif predicted_price == current_price or actual_price == current_price: 
                    correct_dir = 0.5
                else:
                    correct_dir = 0.0

                percent_away_list.append(p_diff)
                correct_direction_list.append(correct_dir)

                print("*", end='')
                sys.stdout.flush()

            print("]")
            sys.stdout.flush()

            day_took = (time.time() - time_s)
            print("Day " + str(days_done) + " of " + str(total_days) + " took " + str(round(day_took / 60, 2)) + " minutes.")
            time_so_far += day_took

            days_done += 1
            test_days -= 1

            t_year, t_month, t_day = get_year_month_day(current_date)

            f = open(real_test_directory + "/" + "SAVE-" + test_name + ".txt", "w")
            f.write("total_days:" + str(total_days) + "\n")
            f.write("days_done:" + str(days_done) + "\n")
            f.write("test_days:" + str(test_days) + "\n")
            f.write("time_so_far:" + str(time_so_far) + "\n")
            f.write("test_year:" + str(t_year) + "\n")
            f.write("test_month:" + str(t_month) + "\n")
            f.write("test_day:" + str(t_day) + "\n")
            f.write("percent_away_list:" + str(percent_away_list) + "\n")
            f.write("correct_direction_list:" + str(correct_direction_list) + "\n")
            f.write("epochs_list:" + str(epochs_list))
            f.close()

        except KeyboardInterrupt:
                print("I acknowledge that you want this to stop.")
                print("Thy will be done.")
                sys.exit(-1)

        except Exception:
            error_handler(symbol, Exception)

    test_year, test_month, test_day = get_year_month_day(current_date)

    print(percent_away_list)
    print(correct_direction_list)
    avg_p = str(round(mean(percent_away_list), 2))
    avg_d = str(round(mean(correct_direction_list) * 100, 2))
    avg_e = str(round(mean(epochs_list), 2))
    print("Parameters: N_steps: " + str(params["N_STEPS"]) + ", Lookup Step:" + str(params["LOOKUP_STEP"]) + ", Test Size: " + str(params["TEST_SIZE"]) + ",")
    print("N_layers: " + str(params["N_LAYERS"]) + ", Cell: " + str(params["CELL"]) + ",")
    print("Units: " + str(params["UNITS"]) + "," + " Dropout: " + str(params["DROPOUT"]) + ", Bidirectional: " + str(params["BIDIRECTIONAL"]) + ",")
    print("Loss: " + params["LOSS"] + ", Optimizer: " + 
    params["OPTIMIZER"] + ", Batch_size: " + str(params["BATCH_SIZE"]) + ",")
    print("Epochs: " + str(params["EPOCHS"]) + ", Patience: " + str(params["PATIENCE"]) + ", Limit: " + str(params["LIMIT"]) + ".")
    print("Feature Columns: " + str(params["FEATURE_COLUMNS"]) + "\n\n")

    print("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent")
    print("and it predicted the correct direction " + avg_d + " percent of the time ")
    print("while using an average of " + avg_e + " epochs.")
    print("The end day was: " + str(test_month) + "-" + str(test_day) + "-" + str(test_year))

    real_test_excel(real_test_excel, test_name, test_year, test_month, test_day, params["N_STEPS"], params["LOOKUP_STEP"], params["TEST_SIZE"], params["N_LAYERS"], 
        params["CELL"], params["UNITS"], params["DROPOUT"], params["BIDIRECTIONAL"], params["LOSS"], params["OPTIMIZER"], params["BATCH_SIZE"],
         params["EPOCHS"], params["PATIENCE"], params["LIMIT"], params["FEATURE_COLUMNS"], avg_p, avg_d, avg_e, time_so_far, total_days)
    print("Testing all of the days took " + str(time_so_far // 3600) + " hours and " + str(round((time_so_far % 60), 2)) + " minutes.")

    if os.path.isfile(real_test_directory + "/" + "SAVE-" + test_name + ".txt"):
        os.remove(real_test_directory + "/" + "SAVE-" + test_name + ".txt")

    delete_files_in_folder(model_saveload_directory + "/" + params["SAVE_FOLDER"])

if __name__ == "__main__":
    # needed to add this line because otherwise the batch run module would get an extra unwanted test
    check_directories()

    params = {
        "N_STEPS": 300,
        "LOOKUP_STEP": 1,
        "TEST_SIZE": 0.2,
        "N_LAYERS": 2,
        "CELL": LSTM,
        "UNITS": 256,
        "DROPOUT": 0.4,
        "BIDIRECTIONAL": False,
        "LOSS": "huber_loss",
        "OPTIMIZER": "adam",
        "BATCH_SIZE": 256,
        "EPOCHS": 800,
        "PATIENCE": 200,
        "SAVELOAD": True,
        "LIMIT": 4000,
        "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
        "SAVE_FOLDER": "batch1"
    }

    the_real_test(test_year, test_month, test_day, test_days, params)
