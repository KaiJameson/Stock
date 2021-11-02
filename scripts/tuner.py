from functions import check_directories, delete_files_in_folder, get_correct_direction, silence_tensorflow, get_test_name
silence_tensorflow()
from tensorflow.keras.layers import LSTM, GRU, Dense, SimpleRNN
from paca_model import saveload_neural_net
from functions.paca_model_functs import (get_api, create_model, get_all_accuracies, predict,  
return_real_predict, load_model_with_data)
from config.symbols import tune_sym_dict, tune_year, tune_month, tune_day, tune_days
from io_functs import  backtest_excel, save_to_dictionary, read_saved_contents, print_backtest_results, comparator_results_excel
from functions.time_functs import increment_calendar, get_actual_price
from functions.error_functs import error_handler
from functions.tuner_functs import grab_index, change_params, get_user_input, update_money
from config.environ import back_test_days, test_var, directory_dict, test_money
from functions.time_functs import get_short_end_date, get_year_month_day
from statistics import mean
import time
import sys
import os
import datetime

check_directories()

master_params = {
    "N_STEPS": [100],
    "LOOKUP_STEP": 1,
    "TEST_SIZE": 0.2,
    "LAYERS": [(256, LSTM), (256, LSTM)],
    "UNITS": [256],
    "DROPOUT": [.8],
    "BIDIRECTIONAL": False,
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "BATCH_SIZE": 1024,
    "EPOCHS": [2000],
    "PATIENCE": [200],
    "SAVELOAD": True,
    "LIMIT": [4000],
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "tuning1"
}


api = get_api()
    
tune_symbols = get_user_input(tune_sym_dict, master_params)

print("\nStaring tuner.py using these following symbols: " + str(tune_symbols) + "\n")

for symbol in tune_symbols:
    index_dict = {
        "n_step_in": 0,
        "unit_in": 0,
        "drop_in": 0,
        "epochs_in": 0,
        "patience_in": 0,
        "limit_in": 0
    }

    still_running = True

    if os.path.isfile(directory_dict["tuning_dir"] + "/" + symbol + "-status.txt"):
        print("A tuning was in process")
        print("pulling info now")

        index_dict  = read_saved_contents(directory_dict["tuning_dir"] + "/" + symbol + "-status.txt", index_dict)

    
    while still_running:
        params = change_params(index_dict, master_params)

        model_name = (symbol + "-" + get_test_name(params))

        progress = {
            "total_days": tune_days,
            "days_done": 1,
            "time_so_far": 0.0,
            "tune_year": tune_year,
            "tune_month": tune_month,
            "tune_day": tune_day,
            "current_money": test_money,
            "percent_away_list": [],
            "correct_direction_list": [],
            "epochs_list": []
        }

        print(model_name)
        print(f"year:{tune_year} month:{tune_month} day:{tune_day}")
        starting_day_price = get_actual_price((get_short_end_date(tune_year, tune_month, tune_day) - datetime.timedelta(1)), 
            api, symbol)
        print(starting_day_price)

        if os.path.isfile(directory_dict["tuning_dir"] + "/" + model_name + ".txt"):
            print("A fully completed file with the name " + model_name + " already exists.")
            print("Exiting this instance of exhaustive tune now: ")
            index_dict = grab_index(index_dict, master_params)
            if index_dict["n_step_in"] == len(master_params["N_STEPS"]):
                still_running = False
                break
            else:
                continue
    
        # check if we already have a save file, if we do, extract the info and run it
        if os.path.isfile(directory_dict["tuning_dir"] + "/" + "SAVE-" + model_name + ".txt"):
            progress = read_saved_contents(directory_dict["tuning_dir"] + "/" + "SAVE-" + model_name + ".txt", progress)
       
        current_date = get_short_end_date(progress["tune_year"], progress["tune_month"], progress["tune_day"])
        try:
            while progress["days_done"] <= progress["total_days"]:
                time_s = time.time()
                current_date = increment_calendar(current_date, api, symbol)
                print("\nCurrently on day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) + " using folder: " + params["SAVE_FOLDER"] + ".\n")
                epochs_run = saveload_neural_net(symbol, current_date, params)
                progress["epochs_list"].append(epochs_run)
                
                # setup to allow the rest of the values to be calculated
                data, model = load_model_with_data(symbol, current_date, params, directory_dict["model_dir"], model_name)

                # first grab the current price by getting the latest value from the og data frame
                y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var]) 
                real_y_values = y_real[-back_test_days:]
                current_price = real_y_values[-1]

                # then use predict fuction to get predicted price
                predicted_price = predict(model, data, params["N_STEPS"])

                # get the actual price for the next day the model tried to predict by incrementing the calendar by one day
                actual_price = get_actual_price(current_date, api, symbol)

                # get the percent difference between prediction and actual
                p_diff = round((abs(actual_price - predicted_price) / actual_price) * 100, 2)

                correct_dir = get_correct_direction(predicted_price, current_price, actual_price)

                progress["percent_away_list"].append(p_diff)
                progress["correct_direction_list"].append(correct_dir)

                day_took = (time.time() - time_s)
                print("Day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) + " took " 
                + str(round(day_took / 60, 2)) + " minutes.", flush=True)

                progress["current_money"] = update_money(progress["current_money"], predicted_price, 
                    current_price, actual_price)
                progress["time_so_far"] += day_took
                progress["days_done"] += 1

                progress["tune_year"], progress["tune_month"], progress["tune_day"] = get_year_month_day(current_date)

                save_to_dictionary(directory_dict["tuning_dir"] + "/" + "SAVE-" + 
                    model_name + ".txt", progress)

            print("Percent away: " + str(progress["percent_away_list"]))
            print("Correct direction %: " + str(progress["correct_direction_list"]))
            avg_p = str(round(mean(progress["percent_away_list"]), 2))
            avg_d = str(round(mean(progress["correct_direction_list"]) * 100, 2))
            avg_e = str(round(mean(progress["epochs_list"]), 2))
            hold_money = round(test_money * (current_price / starting_day_price), 2)

            comparator_results_excel(data, tune_days, directory_dict["tuning_dir"], symbol)
            
            print_backtest_results(params, progress["total_days"], avg_p, avg_d, avg_e, progress["tune_year"], progress["tune_month"], 
                progress["tune_day"], progress["time_so_far"], progress["current_money"], hold_money)
            backtest_excel(directory_dict["tuning_dir"], model_name, progress["tune_year"], progress["tune_month"], progress["tune_day"], 
                params, avg_p, avg_d, avg_e, progress["time_so_far"], progress["total_days"], progress["current_money"], hold_money)

            if os.path.isfile(directory_dict["tuning_dir"] + "/" + "SAVE-" + model_name + ".txt"):
                os.remove(directory_dict["tuning_dir"] + "/" + "SAVE-" + model_name + ".txt")

            delete_files_in_folder(directory_dict["model_dir"] + "/" + params["SAVE_FOLDER"])

            index_dict = grab_index(index_dict, master_params)

            if index_dict["n_step_in"] == len(master_params["N_STEPS"]):
                still_running = False
                print("Ending running the stuff for " + symbol)
                if os.path.isfile(directory_dict["tuning_dir"] + "/" + symbol + "-status.txt"):
                    os.remove(directory_dict["tuning_dir"] + "/" + symbol + "-status.txt")
            else:
                save_to_dictionary(directory_dict["tuning_dir"] + "/" + symbol + "-status", index_dict)


        except KeyboardInterrupt:
                    print("I acknowledge that you want this to stop.")
                    print("Thy will be done.")
                    sys.exit(-1)

        except Exception:
            current_date -= datetime.timedelta(1)
            error_handler(symbol, Exception)


