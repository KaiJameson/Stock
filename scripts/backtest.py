
from config.silen_ten import silence_tensorflow
silence_tensorflow()
from functions.functions import check_directories,  delete_files_in_folder, get_model_name, get_correct_direction
from config.symbols import real_test_symbols, test_year, test_month, test_day, test_days
from config.environ import directory_dict, test_var, back_test_days
from functions.io_functs import backtest_excel,  read_saved_contents, save_to_dictionary, print_backtest_results, graph_epochs_relationship
from functions.time_functs import get_past_datetime, get_year_month_day, increment_calendar, get_actual_price
from functions.error_functs import error_handler
from functions.paca_model_functs import get_api, predict, load_model_with_data, return_real_predict
from paca_model import nn_train_save, configure_gpu
from statistics import mean
from tensorflow.keras.layers import LSTM
import sys
import time
import os


def back_testing(test_year, test_month, test_day, test_days, params):
    configure_gpu()

    symbol = real_test_symbols[0]
    api = get_api()
    
    test_name = get_model_name(params)

    progress = {
        "total_days": test_days,
        "days_done": 1,
        "time_so_far": 0.0,
        "test_year": test_year,
        "test_month": test_month,
        "test_day": test_day,
        "percent_away_list": [],
        "correct_direction_list": [],
        "epochs_list": []
    }

    print(test_name)

    if os.path.isfile(directory_dict["backtest_dir"] + "/" + test_name + ".txt"):
        print("A fully completed file with the name " + test_name + " already exists.")
        print("Exiting the_real_test now: ")
        return

    # check if we already have a save file, if we do, extract the info and run it
    if os.path.isfile(directory_dict["backtest_dir"] + "/" + "SAVE-" + test_name + ".txt"):
        print("A save file was found, opening now: ")
        progress = read_saved_contents(directory_dict["backtest_dir"] + "/" + "SAVE-" + test_name + ".txt", progress)

        
    current_date = get_past_datetime(progress["test_year"], progress["test_month"], progress["test_day"])

    try:
        while progress["days_done"] <= progress["total_days"]:
            time_s = time.time()
            current_date = increment_calendar(current_date, api, symbol)

            for symbol in real_test_symbols:
                print("\nCurrently on day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) + " using folder: " + params["SAVE_FOLDER"] + ".\n")
                epochs_run = nn_train_save(symbol, current_date, params)
                progress["epochs_list"].append(epochs_run)
                
            print("Model result progress: [", end="")
            for symbol in real_test_symbols:
                # get model name for future reference
                model_name = (symbol + "-" + get_model_name(params))

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

                print("*", end="", flush=True)
                
            print("]", flush=True)
            

            day_took = (time.time() - time_s)
            print("Day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) + " took " + str(round(day_took / 60, 2)) + " minutes.")
            progress["time_so_far"] += day_took

            progress["days_done"] += 1

            progress["test_year"], progress["test_month"], progress["test_day"] = get_year_month_day(current_date)

            save_to_dictionary(directory_dict["backtest_dir"] + "/" + "SAVE-" + test_name + ".txt", progress)

        print("Percent away: " + str(progress["percent_away_list"]))
        print("Correct direction %: " + str(progress["correct_direction_list"]))
        avg_p = str(round(mean(progress["percent_away_list"]), 2))
        avg_d = str(round(mean(progress["correct_direction_list"]) * 100, 2))
        avg_e = str(round(mean(progress["epochs_list"]), 2))


        print_backtest_results(params, progress["total_days"], avg_p, avg_d, avg_e, progress["test_year"], progress["test_month"], 
                    progress["test_day"], progress["time_so_far"], None, None)
        backtest_excel(directory_dict["backtest_dir"], test_name, progress["test_year"], progress["test_month"], 
                    progress["test_day"], params, avg_p, avg_d, 
            avg_e, progress["time_so_far"], progress["total_days"], None, None)

        graph_epochs_relationship(progress, test_name)

        if os.path.isfile(directory_dict["backtest_dir"] + "/" + "SAVE-" + test_name + ".txt"):
            os.remove(directory_dict["backtest_dir"] + "/" + "SAVE-" + test_name + ".txt")

        delete_files_in_folder(directory_dict["model_dir"] + "/" + params["SAVE_FOLDER"])

    except KeyboardInterrupt:
            print("I acknowledge that you want this to stop.")
            print("Thy will be done.")
            sys.exit(-1)

    except Exception:
        error_handler(symbol, Exception)


if __name__ == "__main__":
    # needed to add this line because otherwise the batch run module would get an extra unwanted test
    check_directories()

    params = {
        "N_STEPS": 100,
        "LOOKUP_STEP": 1,
        "TEST_SIZE": 0.2,
        "LAYERS": [(256, LSTM), (256, LSTM)],
        "UNITS": 256,
        "DROPOUT": 0.4,
        "BIDIRECTIONAL": False,
        "LOSS": "huber_loss",
        "OPTIMIZER": "adam",
        "BATCH_SIZE": 1024,
        "EPOCHS": 2000,
        "PATIENCE": 200,
        "LIMIT": 4000,
        "SAVELOAD": True,
        "FEATURE_COLUMNS": ["close", "ht_trendmode"],
        "SAVE_FOLDER": "batch1"
    }

    back_testing(test_year, test_month, test_day, test_days, params)
