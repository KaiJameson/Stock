
from config.silen_ten import silence_tensorflow
silence_tensorflow()
from config.symbols import real_test_symbols, test_year, test_month, test_day, test_days
from config.environ import directory_dict, back_test_days
from functions.functions import check_directories, delete_files_in_folder, get_correct_direction, get_test_name, sr2, sr1002, get_model_name
from functions.io_functs import backtest_excel, read_saved_contents, save_to_dictionary, print_backtest_results, graph_epochs_relationship
from functions.time_functs import get_past_datetime, get_year_month_day, increment_calendar, get_actual_price
from functions.error_functs import error_handler
from functions.trade_functs import get_api
from paca_model import ensemble_predictor
from paca_model import configure_gpu
from statistics import mean
from tensorflow.keras.layers import LSTM
import sys
import time
import os


def back_testing(test_year, test_month, test_day, test_days, params):
    configure_gpu()

    symbol = real_test_symbols[0]
    api = get_api()
    
    test_name = get_test_name(params)

    progress = {
        "total_days": test_days,
        "days_done": 1,
        "time_so_far": 0.0,
        "test_year": test_year,
        "test_month": test_month,
        "test_day": test_day,
        "percent_away_list": [],
        "correct_direction_list": [],
        "epochs_dict": {}
    }

    for predictor in params["ENSEMBLE"]:
        if "nn" in predictor:
            progress["epochs_dict"][predictor] = []

    print(test_name)

    if os.path.isfile(directory_dict["backtest"] + "/" + test_name + ".txt"):
        print("A fully completed file with the name " + test_name + " already exists.")
        print("Exiting the_real_test now: ")
        return

    # check if we already have a save file, if we do, extract the info and run it
    if os.path.isfile(directory_dict["backtest"] + "/" + "SAVE-" + test_name + ".txt"):
        print("A save file was found, opening now: ")
        progress = read_saved_contents(directory_dict["backtest"] + "/" + "SAVE-" + test_name + ".txt", progress)

        
    current_date = get_past_datetime(progress["test_year"], progress["test_month"], progress["test_day"])

    try:
        while progress["days_done"] <= progress["total_days"]:
            time_s = time.perf_counter()
            current_date = increment_calendar(current_date, api, symbol)

            for symbol in real_test_symbols:
                print("\nCurrently on day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) + " using folder: " + params["SAVE_FOLDER"] + ".\n")

                predicted_price, current_price, epochs_run = ensemble_predictor(symbol, params, current_date)
                if bool(epochs_run):
                    for predictor in epochs_run:
                        progress["epochs_dict"][predictor].append(epochs_run[predictor])

                # get the actual price for the next day the model tried to predict by incrementing the calendar by one day
                actual_price = get_actual_price(current_date, api, symbol)
                # get the percent difference between prediction and actual
                p_diff = round((abs(actual_price - predicted_price) / actual_price) * 100, 2)

                correct_dir = get_correct_direction(predicted_price, current_price, actual_price)

                progress["percent_away_list"].append(p_diff)
                progress["correct_direction_list"].append(correct_dir)
            

            day_took = (time.perf_counter() - time_s)
            print("Day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) + " took " + str(round(day_took / 60, 2)) + " minutes.", flush=True)
            progress["time_so_far"] += day_took

            progress["days_done"] += 1

            progress["test_year"], progress["test_month"], progress["test_day"] = get_year_month_day(current_date)

            save_to_dictionary(directory_dict["backtest"] + "/" + "SAVE-" + test_name + ".txt", progress)
            for predictor in params["ENSEMBLE"]:
                    if "nn" in predictor: 
                        nn_name = get_model_name(params[predictor])
                        save_to_dictionary(f"""{directory_dict["save_predicts"]}/{nn_name}.txt""", params[predictor]["SAVE_PRED"])

        print("Percent away: " + str(progress["percent_away_list"]))
        print("Correct direction %: " + str(progress["correct_direction_list"]))
        avg_p = sr2(mean(progress["percent_away_list"]))
        avg_d = sr1002(mean(progress["correct_direction_list"]))
        avg_e = {}
        for predictor in params["ENSEMBLE"]:
            if "nn" in predictor:
                avg_e[predictor] = mean(progress["epochs_dict"][predictor])
        

        print_backtest_results(params, progress["total_days"], avg_p, avg_d, avg_e, progress["test_year"], progress["test_month"], 
                    progress["test_day"], progress["time_so_far"], None, None)
        backtest_excel(directory_dict["backtest"], test_name, progress["test_year"], progress["test_month"], 
                    progress["test_day"], params, avg_p, avg_d, 
            avg_e, progress["time_so_far"], progress["total_days"], None, None)

        # graph_epochs_relationship(progress, test_name)

        if os.path.isfile(directory_dict["backtest"] + "/" + "SAVE-" + test_name + ".txt"):
            os.remove(directory_dict["backtest"] + "/" + "SAVE-" + test_name + ".txt")

        delete_files_in_folder(directory_dict["model"] + "/" + params["SAVE_FOLDER"])

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
        "FEATURE_COLUMNS": ["c", "ht_trendmode"],
        "SAVE_FOLDER": "batch1"
    }

    back_testing(test_year, test_month, test_day, test_days, params)
