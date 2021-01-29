from tensorflow.keras.layers import LSTM
from time_functions import get_short_end_date, get_year_month_day
from functions import check_directories, real_test_excel, real_test_directory, delete_files, error_handler, interwebz_pls
from symbols import real_test_symbols, test_year, test_month, test_day, test_days
from alpaca_nn_functions import get_api, create_model, get_all_accuracies, predict, load_data, return_real_predict
from alpaca_neural_net import saveload_neural_net
from environment import error_file, model_saveload_directory, test_var, back_test_days
from statistics import mean
import pandas as pd
import datetime
import sys
import time
import os
import ast


feature_columns = ["open", "low", "high", "close", "mid", "volume"]

n_steps = 300
lookup_step = 1
test_size = .2
n_layers = 2
cell = LSTM
units = 256
dropout = 0.4
bidirectional = False
loss = "huber_loss"
optimizer = "adam"
batch_size = 256
epochs = 800
patience = 200
saveload = True
limit = 4000

def the_real_test(test_year, test_month, test_day, test_days, feature_columns, n_steps, lookup_step, 
    test_size, n_layers, cell, units, dropout, bidirectional, loss, optimizer, batch_size, epochs, 
    patience, saveload, limit):
    days_done = 1

    symbol = real_test_symbols[0]
    api = get_api()
    
    current_date = get_short_end_date(test_year, test_month, test_day)

    test_name = f"{feature_columns}-limit-{limit}-n_step-{n_steps}-layers-{n_layers}-units-{units}-epochs-{epochs}"
    total_days = test_days
    total_tests = len(real_test_symbols) * total_days
    time_so_far = 0.0
    percent_away_list = []
    correct_direction_list = []
    epochs_list = []
    time_ss = time.time()

    if os.path.isfile(real_test_directory + "/" + test_name + ".txt"):
        print("A fully completed file with the name " + test_name + " already exists.")
        print("Exiting the_real_test now: ")
        return

    # check if we already have a save file, if we do, extract the info and run it
    if os.path.isfile(real_test_directory + "/" + "SAVE-" + test_name + ".txt"):
        f = open(real_test_directory + "/" + "SAVE-" + test_name + ".txt", "r")

        file_contents = {}
        for line in f:
            parts = line.strip().split(":")
            file_contents[parts[0]] = parts[1]

        total_days = int(file_contents["total_days"])
        days_done = int(file_contents["days_done"])
        test_days = int(file_contents["test_days"])
        time_so_far = float(file_contents["time_so_far"])
        test_year = int(file_contents["test_year"])
        test_month = int(file_contents["test_month"])
        test_day = int(file_contents["test_day"])
        percent_away_list = ast.literal_eval(file_contents["percent_away_list"])
        correct_direction_list = ast.literal_eval(file_contents["correct_direction_list"])
        epochs_list = ast.literal_eval(file_contents["epochs_list"])
        f.close()

        print("\nOpening an exising test file that was on day " + str(days_done) + " of " + str(total_days) + ".")
        print("It is using these parameters: " + test_name + ".\n")

        current_date = get_short_end_date(test_year, test_month, test_day)


    while test_days > 0:
        try:
            time_s = time.time()
            interwebz_pls("NA", current_date, "calendar")
            calendar = api.get_calendar(start=current_date, end=current_date)[0]
            if calendar.date != current_date:
                print("Skipping " + str(current_date) + " because it was not a market day.")
                current_date = current_date + datetime.timedelta(1)
                continue

            print("\nMoving forward one day in time: \n")

            current_date = current_date + datetime.timedelta(1)

            for symbol in real_test_symbols:
                print("\nCurrently on day " + str(days_done) + " of " + str(total_days) + ".\n")
                epochs_run = saveload_neural_net(symbol, current_date, n_steps, lookup_step, test_size, n_layers, cell, units, dropout,
                bidirectional, loss, optimizer, batch_size, epochs, patience, saveload, limit, feature_columns)
                epochs_list.append(epochs_run)
                
            for symbol in real_test_symbols:
                # setup to allow the rest of the values to be calculated
                data, train, valid, test = load_data(symbol, current_date, n_steps, batch_size, limit, feature_columns, False, to_print=False)
                model = create_model(n_steps, units, cell, n_layers, dropout, loss, optimizer, bidirectional)
                model.load_weights(model_saveload_directory + "/" + symbol + ".h5")

                # first grab the current price by getting the latest value from the og data frame
                y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var]) 
                real_y_values = y_real[-back_test_days:]
                current_price = real_y_values[-1]

                # then use predict fuction to get predicted price
                predicted_price = predict(model, data, n_steps)

                # get the actual price for the next day the model tried to predict by incrementing the calandar by one day
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

    print(percent_away_list)
    print(correct_direction_list)
    avg_p = str(round(mean(percent_away_list), 2))
    avg_d = str(round(mean(correct_direction_list) * 100, 2))
    avg_e = str(round(mean(epochs_list), 2))
    print("Parameters: n_steps: " + str(n_steps) + ", lookup step:" + str(lookup_step) + ", test size: " + str(test_size) + ",")
    print("N_layers: " + str(n_layers) + ", Cell: " + str(cell) + ",")
    print("Units: " + str(units) + "," + " Dropout: " + str(dropout) + ", Bidirectional: " + str(bidirectional) + ",")
    print("Loss: " + loss + ", Optimizer: " + optimizer + ", Batch_size: " + str(batch_size) + ",")
    print("Epochs: " + str(epochs) + ", Patience: " + str(patience) + ", Limit: " + str(limit) + ".")
    print("Feature Columns: " + str(feature_columns) + "\n\n")

    print("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent")
    print("and it predicted the correct direction " + avg_d + " percent of the time ")
    print("while using an average of " + avg_e + " epochs.")
    print("The start day was: " + str(test_month) + "-" + str(test_day) + "-" + str(test_year))

    real_test_excel(test_year, test_month, test_day, n_steps, lookup_step, test_size, n_layers, cell, units, 
        dropout, bidirectional, loss, optimizer, batch_size, epochs, patience, limit, feature_columns, avg_p, 
        avg_d, avg_e, time_so_far, total_days)
    print("Testing all of the days took " + str(time_so_far // 3600) + " hours and " + str(round((time_so_far % 60), 2)) + " minutes.")

    if os.path.isfile(real_test_directory + "/" + "SAVE-" + test_name + ".txt"):
        os.remove(real_test_directory + "/" + "SAVE-" + test_name + ".txt")

if __name__ == "__main__":
    # needed to add this line because otherwise the batch run module would get an extra unwanted test
    check_directories()

    the_real_test(test_year, test_month, test_day, test_days, feature_columns, n_steps,
        lookup_step,test_size, n_layers, cell, units, dropout, bidirectional, loss, 
        optimizer, batch_size, epochs, patience, saveload, limit)
