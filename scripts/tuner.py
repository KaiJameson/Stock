from config.silen_ten import silence_tensorflow
silence_tensorflow()
from config.symbols import tune_sym_dict, tune_year, tune_month, tune_day, tune_days
from config.environ import directory_dict, test_money
from config.model_repository import models
from functions.functions import check_directories, get_correct_direction, get_test_name, sr2, sr1002, r2, get_model_name
from functions.trade_functs import get_api
from functions.io_functs import  backtest_excel, save_to_dictionary, read_saved_contents, print_backtest_results, comparator_results_excel
from functions.time_functs import increment_calendar, get_actual_price, get_calendar
from functions.error_functs import error_handler, keyboard_interrupt
from functions.tuner_functs import subset_and_predict, get_user_input
from functions.compar_functs import update_money
from functions.data_load_functs import df_subset, get_proper_df
from functions.time_functs import get_past_datetime, get_year_month_day
from paca_model import configure_gpu
from make_excel import make_tuning_sheet
from statistics import mean
import pandas as pd
import time
import datetime
import os


def tuning(tune_year, tune_month, tune_day, tune_days, params, output=False):
    api = get_api()
    configure_gpu()
        
    tune_symbols, params = get_user_input(tune_sym_dict, params)

    print("\nStaring tuner.py using these following symbols: " + str(tune_symbols) + "\n")

    output_list = []
    for symbol in tune_symbols:
        test_name = (symbol + "-" + get_test_name(params))

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
            "epochs_dict": {}
        }

        if os.path.isfile(directory_dict["tuning"] + "/" + test_name + ".txt"):
            if output:
                output_dict = {
                    "percent_away": 0.0,
                    "correct_direction": 0.0,
                    "epochs": 0,
                    "total_money": 0.0,
                    "time_so_far": 0.0,
                }

                output_dict = read_saved_contents(f"""{directory_dict["tuning"]}/{test_name}.txt""", output_dict)
                output_list.append([test_name, output_dict["percent_away"], output_dict["correct_direction"],
                    output_dict["time_so_far"], output_dict["total_money"]])
            else:
                print("A fully completed file with the name " + test_name + " already exists.")
                print("Exiting this instance of tuning now: ")
            continue


        for predictor in params["ENSEMBLE"]:
            if "nn" in predictor:
                progress["epochs_dict"][predictor] = []

        print(test_name)
        print(f"year:{tune_year} month:{tune_month} day:{tune_day}")

        master_df = get_proper_df(symbol, params["LIMIT"], "V2")
        tmp_cal = get_calendar(get_past_datetime(tune_year, tune_month, tune_day), api, symbol)
        starting_day_price = get_actual_price((get_past_datetime(tune_year, tune_month, tune_day) 
            - datetime.timedelta(1)), master_df, tmp_cal)

    
        # check if we already have a save file, if we do, extract the info and run it
        if os.path.isfile(directory_dict["tuning"] + "/" + "SAVE-" + test_name + ".txt"):
            progress = read_saved_contents(directory_dict["tuning"] + "/" + "SAVE-" + test_name + ".txt", progress)
    
        current_date = get_past_datetime(progress["tune_year"], progress["tune_month"], progress["tune_day"])
        print(f" starting day price {starting_day_price}")
        calendar = get_calendar(current_date, api, symbol)
        try:
            while progress["days_done"] <= progress["total_days"]:
                time_s = time.perf_counter()

                print("\nCurrently on day " + str(progress["days_done"]) + " of " + str(progress["total_days"]) 
                    + " using folder: " + params["SAVE_FOLDER"] + ".\n")

                current_date = increment_calendar(current_date, calendar)
                predicted_price, current_price, epochs_run, sub_df, data_dict = subset_and_predict(symbol, 
                    params, current_date, master_df)


                if bool(epochs_run):
                    for predictor in epochs_run:
                        progress["epochs_dict"][predictor].append(epochs_run[predictor])

                actual_price = get_actual_price(current_date, master_df, calendar)
                p_diff = round((abs(actual_price - predicted_price) / actual_price) * 100, 2)
                correct_dir = get_correct_direction(predicted_price, current_price, actual_price)
                print(f"Symbol:{symbol} Date:{current_date} Predicted:{sr2(predicted_price)} " 
                    f"Current:{sr2(current_price)} Actual:{sr2(actual_price)} Direction:{correct_dir}\n", flush=True)
                progress["percent_away_list"].append(p_diff)
                progress["correct_direction_list"].append(correct_dir)

                day_took = (time.perf_counter() - time_s)
                print(f"""Day {progress["days_done"]} of {progress["total_days"]} took """ 
                    f"""{r2(day_took / 60)} minutes or {r2(day_took)} seconds.\n""", flush=True)

                progress["current_money"] = update_money(progress["current_money"], predicted_price, 
                    current_price, actual_price)
                progress["time_so_far"] += day_took
                progress["days_done"] += 1

                progress["tune_year"], progress["tune_month"], progress["tune_day"] = get_year_month_day(current_date)

                save_to_dictionary(directory_dict["tuning"] + "/" + "SAVE-" + 
                    test_name + ".txt", progress)
                for predictor in params["ENSEMBLE"]:
                    if "nn" in predictor: 
                        nn_name = get_model_name(params[predictor])
                        for sym in params[predictor]["SAVE_PRED"].copy():
                            if sym != symbol:
                                del params[predictor]["SAVE_PRED"][sym]
                        save_to_dictionary(f"""{directory_dict["save_predicts"]}/{nn_name}/{symbol}.txt""", params[predictor]["SAVE_PRED"])
                del sub_df, data_dict


            print("Percent away: " + str(progress["percent_away_list"]))
            print("Correct direction %: " + str(progress["correct_direction_list"]))
            avg_p = sr2(mean(progress["percent_away_list"]))
            avg_d = sr1002(mean(progress["correct_direction_list"]))
            avg_e = {}
            for predictor in params["ENSEMBLE"]:
                if "nn" in predictor:
                    avg_e[predictor] = mean(progress["epochs_dict"][predictor])
            hold_money = r2(test_money * (current_price / starting_day_price))

            sub_df = df_subset(current_date, master_df)
            comparator_results_excel(sub_df, tune_days, symbol)
            
            
            print_backtest_results(params, progress["total_days"], avg_p, avg_d, avg_e, progress["tune_year"], progress["tune_month"], 
                progress["tune_day"], progress["time_so_far"], progress["current_money"], hold_money)
            backtest_excel(directory_dict["tuning"], test_name, progress["tune_year"], progress["tune_month"], progress["tune_day"], 
                params, avg_p, avg_d, avg_e, progress["time_so_far"], progress["total_days"], progress["current_money"], hold_money)

            if output:
                output_list.append([test_name, avg_p, avg_d, progress["time_so_far"], progress["current_money"]])

            if os.path.isfile(directory_dict["tuning"] + "/" + "SAVE-" + test_name + ".txt"):
                os.remove(directory_dict["tuning"] + "/" + "SAVE-" + test_name + ".txt")

            if os.path.isfile(f"{directory_dict['model']}/{params['SAVE_FOLDER']}"):
                os.remove(f"{directory_dict['model']}/{params['SAVE_FOLDER']}")
            

            print(f"The name for the test was {get_test_name(params)}")
            del sub_df, master_df

        except KeyboardInterrupt:
            keyboard_interrupt()
        except Exception:
            error_handler(symbol, Exception)

    make_tuning_sheet(get_test_name(params))
    
    if output:
        result_df = pd.DataFrame(output_list, columns=["Model Name", "Average percent", "Average direction", 
            "Time used", "Money Made"])
        result_df["Average percent"] = pd.to_numeric(result_df["Average percent"]).astype("float64")
        result_df["Average direction"] = pd.to_numeric(result_df["Average direction"]).astype("float64")
        return [result_df["Model Name"][0], r2(result_df["Average percent"].mean()), r2(result_df["Average direction"].mean()),
            r2(result_df["Time used"].sum() / 60), r2(result_df["Money Made"].mean())]

if __name__ == "__main__":
    check_directories()
    params = {
        # "ENSEMBLE": ["nn4"],
        "ENSEMBLE": ["MLENS1"],
        "TRADING": False,
        "SAVE_FOLDER": "",
        "LIMIT": 4000,
    }

    for model in params["ENSEMBLE"]:
        if model in models:
            params[model] = models[model]
    print(params)

    tuning(tune_year, tune_month, tune_day, tune_days, params)



