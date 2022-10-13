from config.silen_ten import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental import AUTOTUNE
from sklearn.model_selection import TimeSeriesSplit
from config.environ import save_logs, directory_dict
from config.symbols import sym_dict, tune_days, tune_year, tune_month, tune_day, time_kfold_dict
from config.model_repository import models
from functions.data_load import get_df_dict, load_all_data, df_subset
from functions.functions import check_directories, get_test_name, get_model_name, r1002, r2, sr2, sr1002
from functions.trade import get_api
from functions.tuner import get_user_input
from functions.paca_model import create_model, return_real_predict, get_accuracy, get_percent_away
from functions.time import increment_calendar, get_calendar, get_past_datetime, get_year_month_day
from functions.error import error_handler, keyboard_interrupt
from paca_model import configure_gpu, nn_train_save
from make_excel import make_tuning_sheet
from statistics import mean
import numpy as np
import os
import sys
import time


def time_kfold(params):
    configure_gpu()
    api = get_api()

    if len(sys.argv) > 2:
        test_interval = int(sys.argv[2])
        print(test_interval)
        if type(test_interval) != type(1):
            print(f"test_interval must be an int, instead it was {type(test_interval)}")
            print("Please try again")
            sys.exit(-1)
        kfold_symbols, params = get_user_input(sym_dict, params)
    else:
        print("You must give this program two arguments in the style of \"sym#\"")
        print("then \"test_interval\" So that it knows how often to test and what symbols to use.")
        print("Please try again")
        sys.exit(-1)

    

    for symbol in kfold_symbols:
        progress = {
            "total_days": tune_days,
            "days_done": 1,
            "time_so_far": 0.0,
            "tune_year": tune_year,
            "tune_month": tune_month,
            "tune_day": tune_day,
            "current_money": 0,
            "percent_away_list": [],
            "correct_direction_list": [],
            "epochs_list": []
        }

        test_name = f"{symbol}-{test_interval}-{get_test_name(params)}"

        if os.path.isfile(f"{params['TUNE_FOLDER']}/{test_name}.txt"):
            print(f"A fully completed file with the name {test_name} already exists.")
            print("Exiting this instance of tuning now: ")
            continue

        current_date = get_past_datetime(progress['tune_year'], progress['tune_month'], progress['tune_day'])
        calendar = get_calendar(current_date, api, symbol)

        
        accuracies = []
        percents_away = []
        
        predictor = params["ENSEMBLE"][0]
        df_dict = get_df_dict(symbol, params, "V2", True)
        
        try:
            while progress['days_done'] < progress['total_days']:
                time_s = time.perf_counter()
                current_date = increment_calendar(current_date, calendar, test_interval)
                print(f"the current date after increment is {current_date}")
                
                sub_df = df_subset(df_dict, current_date)
                data_dict = load_all_data(params, sub_df, test_interval)

                # print(len(data_dict[predictor]['result']['X_test']))
                # print(len(data_dict[predictor]['result']['y_test']))

                epochs_used = nn_train_save(symbol, params, predictor, data_dict[predictor])
                progress['epochs_list'].append(epochs_used)

                model = create_model(params[predictor])
                model.load_weights(directory_dict["model"] + "/" + params["SAVE_FOLDER"] + "/" + 
                    symbol + "-" + get_model_name(params[predictor]) + ".h5")

                y_real, y_pred = return_real_predict(model, data_dict[predictor]['result']['X_test'], data_dict[predictor]['result']['y_test'],
                     data_dict[predictor]['result']['column_scaler']['future'])

                print(f"len of y_real, y_pred is {len(y_real)}, {len(y_pred)}")

                acc = get_accuracy(y_real, y_pred, lookup_step=1)
                per_away = get_percent_away(y_real, y_pred)
                print(r1002(acc))
                print(r2(per_away))

                progress['percent_away_list'].append(per_away)
                progress['correct_direction_list'].append(acc)

                day_took = (time.perf_counter() - time_s)
                progress['time_so_far'] += day_took
                progress['days_done'] += test_interval
                progress['tune_year'], progress['tune_month'], progress['tune_day'] = get_year_month_day(current_date)

            avg_p = sr2(mean(progress['percent_away_list']))
            avg_d = sr1002(mean(progress['correct_direction_list']))
            avg_e = mean(progress['epochs_list'])


            print(f"{params['TUNE_FOLDER']}/{test_name}")
            with open(f"{params['TUNE_FOLDER']}/{test_name}.txt", "a") as f:
                f.write(f"percent_away|{avg_p}\n")
                f.write(f"correct_direction|{avg_d}\n")
                f.write(f"epochs|{avg_e}\n")
                f.write(f"total_money|{progress['current_money']}\n")
                f.write(f"time_so_far|{progress['time_so_far']}\n")

        except KeyboardInterrupt:
            keyboard_interrupt()
        except Exception:
            error_handler(symbol, Exception)
    
    overall_test_name = f"{test_interval}-{get_test_name(params)}"
    make_tuning_sheet(overall_test_name, params['TUNE_FOLDER'])

if __name__ == "__main__":
    check_directories()
    

    for model in time_kfold_dict["ENSEMBLE"]:
        if model in models:
            time_kfold_dict[model] = models[model]

    print(time_kfold_dict)

    time_kfold(time_kfold_dict)


    
            
                