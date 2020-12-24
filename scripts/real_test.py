from tensorflow.keras.layers import LSTM
from time_functions import get_short_end_date
from functions import check_directories, real_test_excel
from symbols import real_test_symbols, test_year, test_month, test_day, test_days
from alpaca_nn_functions import get_api, create_model, get_all_accuracies, predict, load_data, return_real_predict
from alpaca_neural_net import saveload_neural_net
from environment import error_file, model_saveload_directory, test_var, back_test_days
from statistics import mean
import pandas as pd
import traceback
import datetime
import sys
import time


days_done = 1

check_directories()
api = get_api()
current_date = get_short_end_date(test_year, test_month, test_day)

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
feature_columns = ["open", "low", "high", "close", "mid", "volume"]

total_days = test_days
percent_away_list = []
correct_direction_list = []
time_ss = time.time()

while test_days > 0:
    try:
        calendar = api.get_calendar(start=current_date, end=current_date)[0]
        if calendar.date != current_date:
            print("Skipping " + str(current_date) + " because it was not a market day.")
            current_date = current_date + datetime.timedelta(1)
            continue

        time_s = time.time()
        
        print("\n Moving forward one day in time: \n\n")

        current_date = current_date + datetime.timedelta(1)

        for symbol in real_test_symbols:
            saveload_neural_net(symbol, current_date, n_steps, lookup_step, test_size, n_layers, cell, units, dropout,
            bidirectional, loss, optimizer, batch_size, epochs, patience, saveload, limit, feature_columns)
            
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


        print("Day " + str(days_done) + " of " + str(total_days) + " took " + str((time.time() - time_s) / 60) + " minutes.")

        days_done += 1
        test_days -= 1
    #if the day is a valid market day
    except KeyboardInterrupt:
            print("I acknowledge that you want this to stop.")
            print("Thy will be done.")
            sys.exit(-1)

    except:
        f = open(error_file, "a")
        f.write("problem with configged stock: " + symbol + "\n")
        exit_info = sys.exc_info()
        f.write(str(exit_info[1]) + "\n")
        traceback.print_tb(tb=exit_info[2], file=f)
        f.close()
        print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

print(percent_away_list)
print(correct_direction_list)
avg_p = str(round(mean(percent_away_list), 2))
avg_d = str(round(mean(correct_direction_list) * 100, 2))
print("Parameters: n_steps: " + str(n_steps) + ", lookup step:" + str(lookup_step) + ", test size: " + str(test_size) + ",")
print("N_layers: " + str(n_layers) + ", Cell: " + str(cell) + ",")
print("Units: " + str(units) + "," + " Dropout: " + str(dropout) + ", Bidirectional: " + str(bidirectional) + ",")
print("Loss: " + loss + ", Optimizer: " + optimizer + ", Batch_size: " + str(batch_size) + ",")
print("Epochs: " + str(epochs) + ", Patience: " + str(patience) + ", Limit: " + str(limit) + ".")
print("Feature Columns: " + str(feature_columns) + "\n\n")

print("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent")
print("and it predicted the correct direction " + avg_d + " percent of the time.")

time_taken = round((time.time() - time_ss) / 60, 2)

real_test_excel(n_steps, lookup_step, test_size, n_layers, cell, units, dropout, bidirectional, loss, 
    optimizer, batch_size, epochs, patience, limit, feature_columns, avg_p, avg_d, time_taken, total_days)

print("Testing all of the days took " + str(time_taken) + " minutes.")




    