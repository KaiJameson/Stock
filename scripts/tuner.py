from config.silen_ten import silence_tensorflow
silence_tensorflow()
from config.symbols import sym_dict, tuner_dict, tune_year, tune_month, tune_day, tune_days
from config.environ import directory_dict, test_money
from config.model_repository import models
from functions.functions import check_directories, get_correct_direction, get_test_name, sr2, sr1002, r2, get_model_name
from functions.trade import get_api
from functions.io import  backtest_excel, save_to_dictionary, read_saved_contents, print_backtest_results, comparator_results_excel
from functions.time import increment_calendar, get_actual_price, get_calendar
from functions.error import error_handler, keyboard_interrupt
from functions.tuner import subset_and_predict, get_user_input
from functions.comparators import update_money
from functions.data_load import df_subset, get_df_dict
from functions.time import get_past_datetime, get_year_month_day
from multiprocessing.pool import Pool
from paca_model import configure_gpu
from make_excel import make_tuning_sheet
from statistics import mean
import time
import datetime
import os


def tuning(symbol, tune_year, tune_month, tune_day, tune_days, params, output=False):
    test_name = (symbol + "-" + get_test_name(params))
    output_list = []

    if os.path.isfile(f"{params['TUNE_FOLDER']}/{test_name}.txt"):
        if output:
            output_dict = {
                "percent_away": 0.0,
                "correct_direction": 0.0,
                "epochs": 0.0,
                "total_money": 0.0,
                "time_so_far": 0.0,
            }

            output_dict = read_saved_contents(f"{params['TUNE_FOLDER']}/{test_name}.txt", output_dict)
            output_list = [test_name, output_dict['percent_away'], output_dict['correct_direction'],
                output_dict['time_so_far'], output_dict['total_money']]
            return output_list
        else:
            print(f"A fully completed file with the name {test_name} already exists.")
            print("Exiting this instance of tuning now: ")
            return

    api = get_api()
    
    if not output:
        configure_gpu()



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


    test_var = "c"
    for predictor in params['ENSEMBLE']:
        if "nn" in predictor:
            progress['epochs_dict'][predictor] = []
        if params[predictor]['TEST_VAR'] == "acc":
            test_var = "acc"

    print(test_name)
    print(f"year:{tune_year} month:{tune_month} day:{tune_day}")


    df_dict = get_df_dict(symbol, params, to_print=True)
    tmp_cal = get_calendar(get_past_datetime(tune_year, tune_month, tune_day), api, symbol)
    starting_day_price = get_actual_price((get_past_datetime(tune_year, tune_month, tune_day) 
        - datetime.timedelta(1)), df_dict['price'], tmp_cal)


    # check if we already have a save file, if we do, extract the info and run it
    if os.path.isfile(f"{params['TUNE_FOLDER']}/SAVE-{test_name}.txt"):
        progress = read_saved_contents(f"{params['TUNE_FOLDER']}/SAVE-{test_name}.txt", progress)

    current_date = get_past_datetime(progress['tune_year'], progress['tune_month'], progress['tune_day'])
    print(f"Starting day price:{starting_day_price}")
    calendar = get_calendar(current_date, api, symbol)
    print(progress)
    try:
        while progress['days_done'] <= progress['total_days']:
            time_s = time.perf_counter()

            print(f"\nCurrently on day {progress['days_done']} of {progress['total_days']} "
                f"using ensemble: {params['ENSEMBLE']} with folder:{params['SAVE_FOLDER']}.\n")

            current_date = increment_calendar(current_date, calendar, 1)
            predicted_price, current_price, epochs_run = subset_and_predict(symbol, 
                params, current_date, df_dict)


            if bool(epochs_run):
                for predictor in epochs_run:
                    progress['epochs_dict'][predictor].append(epochs_run[predictor])

            actual_price = get_actual_price(current_date, df_dict['price'], calendar)
            if test_var == "acc":
                p_diff = 0.0
            else:
                p_diff = round((abs(actual_price - predicted_price) / actual_price) * 100, 2)
            correct_dir = get_correct_direction(predicted_price, current_price, actual_price, test_var)
            print(f"Symbol:{symbol} Date:{current_date} Predicted:{sr2(predicted_price)} " 
                f"Current:{sr2(current_price)} Actual:{sr2(actual_price)} Direction:{correct_dir}\n", flush=True)
            progress['percent_away_list'].append(p_diff)
            progress['correct_direction_list'].append(correct_dir)

            day_took = (time.perf_counter() - time_s)
            print(f"Day {progress['days_done']} of {progress['total_days']} took " 
                f"{r2(day_took / 60)} minutes or {r2(day_took)} seconds.\n", flush=True)

            progress['current_money'] = update_money(progress['current_money'], predicted_price, 
                current_price, actual_price, test_var)
            progress['time_so_far'] += day_took
            progress['days_done'] += 1

            progress['tune_year'], progress['tune_month'], progress['tune_day'] = get_year_month_day(current_date)

            save_to_dictionary(f"{params['TUNE_FOLDER']}/SAVE-{test_name}.txt", progress)
            for predictor in params['ENSEMBLE']:
                if "nn" in predictor: 
                    nn_name = get_model_name(params[predictor])
                    for sym in params[predictor]['SAVE_PRED'].copy():
                        if sym != symbol:
                            del params[predictor]['SAVE_PRED'][sym]
                    save_to_dictionary(f"{directory_dict['save_predicts']}/{nn_name}/{symbol}.txt", params[predictor]['SAVE_PRED'])


        print(f"Percent away: {progress['percent_away_list']}")
        print(f"Correct direction %: {progress['correct_direction_list']}")
        avg_p = sr2(mean(progress['percent_away_list']))
        avg_d = sr1002(mean(progress['correct_direction_list']))
        avg_e = {}
        for predictor in params['ENSEMBLE']:
            if "nn" in predictor:
                avg_e[predictor] = mean(progress['epochs_dict'][predictor])
        hold_money = r2(test_money * (current_price / starting_day_price))

        sub_df = df_subset(df_dict, current_date)
        comparator_results_excel(sub_df["price"], tune_days, symbol)
        
        
        print_backtest_results(params, progress, avg_p, avg_d, avg_e, hold_money)
        backtest_excel(params, progress, avg_p, avg_d, avg_e, hold_money, test_name)

        if output:
            output_list = [test_name, avg_p, avg_d, progress["time_so_far"], progress["current_money"]]

        if os.path.isfile(params['TUNE_FOLDER'] + "/" + "SAVE-" + test_name + ".txt"):
            os.remove(params['TUNE_FOLDER'] + "/" + "SAVE-" + test_name + ".txt")

        if os.path.isfile(f"{directory_dict['model']}/{params['SAVE_FOLDER']}"):
            os.remove(f"{directory_dict['model']}/{params['SAVE_FOLDER']}")
        

        print(f"The name for the test was {get_test_name(params)}")
       

    except KeyboardInterrupt:
        keyboard_interrupt()
    except Exception:
            error_handler(symbol, Exception)

    if output:
        print(output_list)
        return output_list
    


if __name__ == "__main__":
    check_directories()
    

    for model in tuner_dict["ENSEMBLE"]:
        if model in models:
            tuner_dict[model] = models[model]
    print(tuner_dict)


    tune_symbols, params = get_user_input(sym_dict, tuner_dict)
    print(params)
    print("\nStaring tuner.py using these following symbols: " + str(tune_symbols) + "\n", flush=True)


    for symbol in tune_symbols:
        with Pool(1) as pool:
            items = [(symbol, tune_year, tune_month, tune_day, tune_days, tuner_dict)]
            for result in pool.starmap(tuning, items):
                pass
        pool.close()
        pool.join()

    make_tuning_sheet(get_test_name(params), params['TUNE_FOLDER'], tune_symbols)



