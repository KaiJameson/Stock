from functions import check_directories, delete_files_in_folder, get_correct_direction, silence_tensorflow, get_test_name
silence_tensorflow()
from tensorflow.keras.layers import LSTM
from paca_model import saveload_neural_net
from paca_model_functs import (get_api, create_model, get_all_accuracies, predict,  
return_real_predict, load_model_with_data)
from symbols import tune_sym_dict, tune_year, tune_month, tune_day, tune_days
from io_functs import  backtest_excel, save_to_dictionary, read_saved_contents, print_backtest_results
from time_functs import increment_calendar, get_current_price
from error_functs import error_handler
from tuner_functs import grab_index, change_params, get_user_input
from environ import back_test_days, test_var, directory_dict
from time_functs import get_short_end_date, get_year_month_day
from statistics import mean
import time
import sys
import os
import traceback
import datetime
import pandas as pd
import ast


check_directories()

master_params = {
    "N_STEPS": [300],
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "N_LAYERS": 2,
    "CELL": LSTM,
    "UNITS": [256],
    "DROPOUT": [.4],
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 256,
    "EPOCHS": [1],
    "PATIENCE": [200],
    "SAVELOAD": True,
    "LIMIT": [4000],
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "tuning1"
}


api = get_api()
    
# total_days = tune_days
tune_symbols = get_user_input(tune_sym_dict, master_params)

print("\nStaring tuner.py using these following symbols: " + str(tune_symbols) + "\n")

for symbol in tune_symbols:
    # n_step_in = unit_in = drop_in = epochs_in = patience_in = limit_in = 0

    index_dict = {
        "n_step_in": 0,
        "unit_in": 0,
        "drop_in": 0,
        "epochs_in": 0,
        "patience_in": 0,
        "limit_in": 0
    }


    still_running = True

    # tune_days = total_days # reset the days count for while loop

    if os.path.isfile(directory_dict["tuning_directory"] + "/" + symbol + "-status.txt"):
        print("A tuning was in process")
        print("pulling info now")

        index_dict  = read_saved_contents(directory_dict["tuning_directory"] + "/" + symbol + "-status.txt", index_dict)

    
    while still_running:
        params = change_params(index_dict, master_params)

        # get model name for future reference
        model_name = (symbol + "-" + get_test_name(params))
        # days_done = 1
        # total_days = tune_days
        # time_so_far = 0.0
        # percent_away_list = []
        # correct_direction_list = []
        # epochs_list = []

        progress = {
            "total_days": tune_days,
            "days_done": 1,
            # "tune_days":
            "time_so_far": 0.0,
            "tune_year": tune_year,
            "tune_month": tune_month,
            "tune_day": tune_day,
            "percent_away_list": [],
            "correct_direction_list": [],
            "epochs_list": []

        }

        print(model_name)

        if os.path.isfile(directory_dict["tuning_directory"] + "/" + model_name + ".txt"):
            print("A fully completed file with the name " + model_name + " already exists.")
            print("Exiting this instance of exhaustive tune now: ")
            index_dict = grab_index(index_dict, master_params)
            if index_dict["n_step_in"] == len(master_params["N_STEPS"]):
                still_running = False
                break
            else:
                continue
    
        # check if we already have a save file, if we do, extract the info and run it
        if os.path.isfile(directory_dict["tuning_directory"] + "/" + "SAVE-" + model_name + ".txt"):
            # total_days, days_done, tune_days, time_so_far, exhaust_year, exhaust_month, exhaust_day, percent_away_list, correct_direction_list, epochs_list = read_saved_contents(directory_dict["tuning_directory"], model_name)
            progress = read_saved_contents(directory_dict["tuning_directory"] + "/" + "SAVE-" + model_name + ".txt", progress)
       
        current_date = get_short_end_date(progress["tune_year"], progress["tune_month"], progress["tune_day"])
        try:
            while progress["days_done"] <= progress["total_days"]:
                time_s = time.time()
                current_date = increment_calendar(current_date, api, symbol)

                print("\nCurrently on day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) + " using folder: " + params["SAVE_FOLDER"] + ".\n")
                epochs_run = saveload_neural_net(symbol, current_date, params)
                progress["epochs_list"].append(epochs_run)
                
                # setup to allow the rest of the values to be calculated
                data, model = load_model_with_data(symbol, current_date, params, directory_dict["model_directory"], model_name)

                # first grab the current price by getting the latest value from the og data frame
                y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var]) 
                real_y_values = y_real[-back_test_days:]
                current_price = real_y_values[-1]

                # then use predict fuction to get predicted price
                predicted_price = predict(model, data, params["N_STEPS"])

                # get the actual price for the next day the model tried to predict by incrementing the calendar by one day
                actual_price = get_current_price(current_date, api, symbol)

                # get the percent difference between prediction and actual
                p_diff = round((abs(actual_price - predicted_price) / actual_price) * 100, 2)

                correct_dir = get_correct_direction(predicted_price, current_price, actual_price)

                progress["percent_away_list"].append(p_diff)
                progress["correct_direction_list"].append(correct_dir)

                day_took = (time.time() - time_s)
                print("Day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) + " took " 
                + str(round(day_took / 60, 2)) + " minutes.")
                progress["time_so_far"] += day_took

                progress["days_done"] += 1
                # tune_days -= 1

                progress["tune_year"], progress["tune_month"], progress["tune_day"] = get_year_month_day(current_date)

                save_to_dictionary(directory_dict["tuning_directory"] + "/" + "SAVE-" + 
                    model_name + ".txt", progress)
                # f = open(directory_dict["tuning_directory"] + "/" + "SAVE-" + model_name + ".txt", "w")
                # f.write("total_days:" + str(total_days) + "\n")
                # f.write("days_done:" + str(days_done) + "\n")
                # f.write("test_days:" + str(tune_days) + "\n")
                # f.write("time_so_far:" + str(time_so_far) + "\n")
                # f.write("exhaust_year:" + str(t_year) + "\n")
                # f.write("exhaust_month:" + str(t_month) + "\n")
                # f.write("exhaust_day:" + str(t_day) + "\n")
                # f.write("percent_away_list:" + str(percent_away_list) + "\n")
                # f.write("correct_direction_list:" + str(correct_direction_list) + "\n")
                # f.write("epochs_list:" + str(epochs_list))
                # f.close()

            # tune_year, tune_month, tune_day = get_year_month_day(current_date)

            print("Percent away: " + str(progress["percent_away_list"]))
            print("Correct direction %: " + str(progress["correct_direction_list"]))
            avg_p = str(round(mean(progress["percent_away_list"]), 2))
            avg_d = str(round(mean(progress["correct_direction_list"]) * 100, 2))
            avg_e = str(round(mean(progress["epochs_list"]), 2))
            
            print_backtest_results(params, progress["total_days"], avg_p, avg_d, avg_e, progress["tune_year"], progress["tune_month"], 
                progress["tune_day"], progress["time_so_far"])
            backtest_excel(directory_dict["tuning_directory"], model_name, progress["tune_year"], progress["tune_month"], progress["tune_day"], 
                params, avg_p, avg_d, avg_e, progress["time_so_far"], progress["total_days"])

            if os.path.isfile(directory_dict["tuning_directory"] + "/" + "SAVE-" + model_name + ".txt"):
                os.remove(directory_dict["tuning_directory"] + "/" + "SAVE-" + model_name + ".txt")

            delete_files_in_folder(directory_dict["model_directory"] + "/" + params["SAVE_FOLDER"])

            index_dict = grab_index(index_dict, master_params)

            if index_dict["n_step_in"] == len(master_params["N_STEPS"]):
                still_running = False
                print("Ending running the stuff for " + symbol)

                # tune_days = total_days # reset the days count for while loop

                if os.path.isfile(directory_dict["tuning_directory"] + "/" + symbol + "-status.txt"):
                    os.remove(directory_dict["tuning_directory"] + "/" + symbol + "-status.txt")
            else:
                save_to_dictionary(directory_dict["tuning_directory"] + "/" + symbol + "-status", index_dict)
                # f = open(directory_dict["tuning_directory"] + "/" + symbol + "-status.txt", "w")
                # f.write("n_step_in:" + str(n_step_in) + "\n")
                # f.write("unit_in:" + str(unit_in) + "\n")
                # f.write("drop_in:" + str(drop_in) + "\n")
                # f.write("epochs_in:" + str(epochs_in) + "\n")
                # f.write("patience_in:" + str(patience_in) + "\n")
                # f.write("limit_in:" + str(limit_in) + "\n")
                # f.close()

                # tune_days = total_days # reset the days count for while loop



        except KeyboardInterrupt:
                    print("I acknowledge that you want this to stop.")
                    print("Thy will be done.")
                    sys.exit(-1)

        except Exception:
            current_date -= datetime.timedelta(1)
            error_handler(symbol, Exception)


