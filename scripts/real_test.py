from tensorflow.keras.layers import LSTM
from time_functions import get_short_end_date
from functions import check_directories
from symbols import real_test_symbols, test_year, test_month, test_day, test_days
from alpaca_nn_functions import get_api, create_model, get_all_accuracies, predict, load_data
from alpaca_neural_net import saveload_neural_net
from environment import error_file, model_saveload_directory
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
batch_size = 64
epochs = 2
patience = 200
saveload = True

time_ss = time.time()

while test_days > 0:
    try:
        current_date = current_date + datetime.timedelta(1)
        
        calendar = api.get_calendar(start=current_date, end=current_date)[0]
        if calendar.date != current_date:
            print("Skipping " + str(current_date) + " because it was not a market day.")
            continue

        time_s = time.time()

        for symbol in real_test_symbols:
            saveload_neural_net(symbol, current_date, n_steps, lookup_step, test_size, n_layers, cell, units, dropout,
            bidirectional, loss, optimizer, batch_size, epochs, patience, saveload)
            
        for symbol in real_test_symbols:
            data, train, valid, test = load_data(symbol, n_steps, True, False)
            # print(data)
            model = create_model(n_steps, units, cell, n_layers, dropout, loss, optimizer, bidirectional)
            model.load_weights(model_saveload_directory + "/" + symbol + ".h5")
            train_acc, valid_acc, test_acc = get_all_accuracies(model, data, lookup_step)
            predicted_price = predict(model, data, n_steps)

            cal = api.get_calendar(start=current_date + datetime.timedelta(1), end=current_date + datetime.timedelta(1))[0]
            print(cal)
            print(cal.date)
            # one_day_in_future = datetime.datetime.fromisocalendar(cal.date)
            
            # one_day_in_future = one_day_in_future.date(cal.date)
            df = api.polygon.historic_agg_v2(symbol, 1, "day", _from=str(cal.date), to=str(cal.date)).df
            print(df)
            actual_price = df["close"].values
            print(actual_price)
            print(predicted_price)


        print("Day " + str(days_done) + " of " + str(test_days) + " took " + str((time.time() - time_s) / 60) + " minutes.")

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


print("Testing all of the days took " + str((time.time() - time_ss) / 60) + " minutes.")

    


