import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.layers import LSTM
from alpaca_neural_net import saveload_neural_net
from alpaca_nn_functions import (get_api, create_model, get_all_accuracies, predict, load_data, 
return_real_predict, load_model_with_data)
from symbols import exhaustive_symbols, exhaust_year, exhaust_month, exhaust_day
from functions import (check_directories, get_short_end_date, interwebz_pls,  get_test_name, 
real_test_excel, delete_files_in_folder, read_saved_contents)
from error_functs import error_handler
from environment import (config_directory, tuning_directory, error_file, make_config, 
back_test_days, tune_days, test_var, model_saveload_directory)
from time_functions import get_short_end_date, get_year_month_day
from statistics import mean
import time
import sys
import os
import traceback
import datetime
import pandas as pd
import ast

ticker = exhaustive_symbols

check_directories()


master_params = {
    "N_STEPS": [50, 100],
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "N_LAYERS": 2,
    "CELL": LSTM,
    "UNITS": [128, 256],
    "DROPOUT": [.35, .4],
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 64,
    "EPOCHS": [800, 1000],
    "PATIENCE": [100, 200],
    "SAVELOAD": True,
    "LIMIT": [2000, 4000],
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume", "7_moving_avg"],
    "SAVE_FOLDER": "tuning1"
}



done_dir = tuning_directory + "/done"
if not os.path.isdir(done_dir):
    os.mkdir(done_dir)
current_dir = tuning_directory + "/current"
if not os.path.isdir(current_dir):
    os.mkdir(current_dir)
f_name = done_dir + "/" + ticker + ".txt"
file_name = current_dir + "/" + ticker + ".csv"
tuning_status_file = tuning_directory + "/" + ticker + ".txt"
if not os.path.isfile(tuning_status_file):
    f = open(tuning_status_file, "w")
    f.close()
done_message = "You are done tuning this stock."

api = get_api()
    
n_step_in = unit_in = drop_in = epochs_in = patience_in = limit_in = 0
still_running = True

for symbol in exhaustive_symbols:
    try:
        if os.path.isfile(tuning_directory + "/" + symbol + "-status.txt"):
            print("A tuning was in process")
            print("pulling info now")

            f = open(tuning_directory + "/" + symbol + "-status.txt")

            file_contents = {}
            for line in f:
                parts = line.strip().split(":")
                file_contents[parts[0]] = parts[1]

            n_step_in = int(file_contents["n_step_in"])
            unit_in = int(file_contents["unit_in"])
            drop_in = int(file_contents["drop_in"])
            epochs_in = int(file_contents["epochs_in"])
            patience_in = int(file_contents["patience_in"])
            limit_in = int(file_contents["limit_in"])
            f.close()

        while still_running:
            params = change_params(n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in, master_params)

            test_name = get_test_name(params)
            total_days = tune_days
            time_so_far = 0.0
            percent_away_list = []
            correct_direction_list = []
            epochs_list = []
            print(test_name)

            if os.path.isfile(tuning_directory + "/" + test_name + ".txt"):
                print("A fully completed file with the name " + test_name + " already exists.")
                print("Exiting the_real_test now: ")
                continue

            # check if we already have a save file, if we do, extract the info and run it
            total_days, days_done, test_days, time_so_far, exhaust_year, exhaust_month, exhaust_day = read_saved_contents(tuning_directory, test_name)

            current_date = get_short_end_date(exhaust_year, exhaust_month, exhaust_day)

            while test_days > 0:
                current_date = increment_calendar(current_date, api, symbol)

                print("\nCurrently on day " + str(days_done) + " of " + str(total_days) + " using folder: " + params["SAVE_FOLDER"] + ".\n")
                epochs_run = saveload_neural_net(symbol, current_date, params)
                epochs_list.append(epochs_run)

                # get model name for future reference
                model_name = (symbol + "-" + get_test_name(params))

                # setup to allow the rest of the values to be calculated
                data, model = load_model_with_data(symbol, current_date, params, tuning_directory, model_name)

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

                day_took = (time.time() - time_s)
                print("Day " + str(days_done) + " of " + str(total_days) + " took " + str(round(day_took / 60, 2)) + " minutes.")
                time_so_far += day_took

                days_done += 1
                test_days -= 1

                t_year, t_month, t_day = get_year_month_day(current_date)

                f = open(tuning_directory + "/" + "SAVE-" + test_name + ".txt", "w")
                f.write("total_days:" + str(total_days) + "\n")
                f.write("days_done:" + str(days_done) + "\n")
                f.write("test_days:" + str(test_days) + "\n")
                f.write("time_so_far:" + str(time_so_far) + "\n")
                f.write("exhaust_year:" + str(t_year) + "\n")
                f.write("exhaust_month:" + str(t_month) + "\n")
                f.write("exhaust_day:" + str(t_day) + "\n")
                f.write("percent_away_list:" + str(percent_away_list) + "\n")
                f.write("correct_direction_list:" + str(correct_direction_list) + "\n")
                f.write("epochs_list:" + str(epochs_list))
                f.close()

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

            real_test_excel(test_year, test_month, test_day, params["N_STEPS"], params["LOOKUP_STEP"], params["TEST_SIZE"], params["N_LAYERS"], 
                params["CELL"], params["UNITS"], params["DROPOUT"], params["BIDIRECTIONAL"], params["LOSS"], params["OPTIMIZER"], params["BATCH_SIZE"],
                params["EPOCHS"], params["PATIENCE"], params["LIMIT"], params["FEATURE_COLUMNS"], avg_p, avg_d, avg_e, time_so_far, total_days)
            print("Testing all of the days took " + str(time_so_far // 3600) + " hours and " + str(round((time_so_far % 60), 2)) + " minutes.")

            if os.path.isfile(tuning_directory + "/" + "SAVE-" + test_name + ".txt"):
                os.remove(tuning_directory + "/" + "SAVE-" + test_name + ".txt")

            delete_files_in_folder(model_saveload_directory + "/" + params["SAVE_FOLDER"])

            n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in, params = grab_index(n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in, params)

            if n_step_in == len(params["N_STEPS"]):
                still_running = False
                # TODO say end of file shit
            else:
                f.open(tuning_directory + "/" + symbol + "-status".txt, "w")
                f.write("n_step_in:" + str(n_step_in) + "\n")
                f.write("unit_in:" + str(unit_in) + "\n")
                f.write("drop_in:" + str(drop_in) + "\n")
                f.write("epochs_in:" + str(epochs_in) + "\n")
                f.write("patience_in:" + str(patience_in) + "\n")
                f.write("limit_in:" + str(limit_in) + "\n")
                f.close()



    except KeyboardInterrupt:
                print("I acknowledge that you want this to stop.")
                print("Thy will be done.")
                sys.exit(-1)

    except Exception:
        if date_changed:
            current_date = current_date - datetime.timedelta(1)
        error_handler(symbol, Exception)

def grab_index(n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in, params):
        while n_step_in < len(params["N_STEPS"]):
            while unit_in < len(params["UNITS"]):
                while drop_in < len(params["DROPOUT"]):
                    while epochs_in < len(params["EPOCHS"]):
                        while patience_in < len(params["PATIENCE"]):
                            while limit_in < len(params["LIMIT"]):
                                limit_in += 1
                                if limit_in < len(params["LIMIT"]):
                                    return n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in
                            patience_in += 1
                            limit_in %= len(params["LIMIT"])
                            if patience_in < len(params["PATIENCE"]):
                                return n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in
                        epochs_in += 1
                        patience_in %= len(params["PATIENCE"])
                        if epochs_in < len(params["EPOCHS"]):
                            return n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in
                    drop_in += 1
                    epochs_in %= len(params["EPOCHS"])
                    if drop_in < len(params["DROPOUT"]):
                        return n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in
                unit_in += 1
                drop_in %= len(params["DROPOUT"])
                if unit_in < len(params["UNITS"]):
                    return n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in
            n_step_in += 1
            unit_in %= len(params["UNITS"])
            if n_step_in < len(params["N_STEPS"]):
                return n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in
        return n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in



def change_params(n_step_in, unit_in, drop_in, epochs_in, patience_in, limit_in, params):
    params["N_STEPS"] =  params[n_step_in]
    params["UNITS"] = params[unit_in]
    params["DROPOUT"] = params[drop_in]
    params["EPOCHS"] = params[epochs_in]
    params["PATIENCE"] = params[patience_in]
    params["LIMIT"] = params[limit_in]

    return params
