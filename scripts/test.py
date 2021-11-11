import math
from config.environ import *
from paca_model import configure_gpu, nn_train_save
from functions.tuner_functs import moving_average_comparator, linear_regression_comparator
from functions.paca_model_functs import *
from functions.data_load_functs import *
from functions.io_functs import *
from functions.functions import *
from functions.time_functs import *
from statistics import mean

def backtest_comparators():
    load_save_symbols = ["AGYS", "AMKR", "BG","BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
        "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]

    over_all_avg = {i:[0.0, 0.0, 0.0]
    for i in range(2, 21)}

    over_all_lin = {i:[0.0, 0.0, 0.0]
    for i in range(2, 21)}
    
    for symbol in load_save_symbols:
        print(symbol, flush=True)
        for i in range(2, 21):
            data, train, valid, test = load_data(symbol, defaults, shuffle=False, to_print=False)
            avg = moving_average_comparator(data, i, 2000)
            lin = linear_regression_comparator(data, i, 2000)
            
            over_all_avg[i][0] += float(avg[0])
            over_all_avg[i][1] += float(avg[1])
            over_all_avg[i][2] += float(avg[2])
            over_all_lin[i][0] += float(lin[0])
            over_all_lin[i][1] += float(lin[1])
            over_all_lin[i][2] += float(lin[2])


    print("Average:")
    for j in range(2, 21):
        print(f"{j}", end="")
        for metric in over_all_avg[j]:
            print(f" {metric / len(load_save_symbols)} ", end="")
        print()

    print("Linear Regression:")
    for j in range(2, 21):
        print(f"{j}", end="")
        for metric in over_all_lin[j]:
            print(f" {metric / len(load_save_symbols)} ", end="")
        print()


if __name__ == "__main__":
    from config.symbols import *

    defaults = {
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
    "EPOCHS": 200,
    "PATIENCE": 200,
    "LIMIT": 4000,
    "SAVELOAD": True,
    "FEATURE_COLUMNS": ["open", "low", "high", "close", "mid", "volume"],
    "SAVE_FOLDER": "tuning4"
    }

    # for symbol in load_save_symbols:
    #     load_data(symbol, defaults, None, False, True, True)


    # Testing for geting scaling/classification to work
    # symbol = "AGYS"
    # nn_train_save(symbol, params=defaults)
    
    # start_time = time.time()
    # model_name = (symbol + "-" + get_model_name(defaults))

    # print("\n~~~Now Starting " + symbol + "~~~")
    
    # time_s = time.time()
    # data, train, valid, test = load_data(symbol, defaults, shuffle=False, to_print=False)
    # print("Loading the data took " + str(time.time() - time_s) + " seconds")    
    # print(f" this is the data: {data}")
    # time_s = time.time()
    # model = create_model(defaults)
    # model.load_weights(directory_dict["model_dir"] + "/" + defaults["SAVE_FOLDER"] + "/" + model_name + ".h5")
    # print("Loading the model took " + str(time.time() - time_s) + " seconds")    

    # time_s = time.time()
    # train_acc, valid_acc, test_acc = get_all_accuracies(model, data, defaults["LOOKUP_STEP"], False)
    # print("Getting the accuracies took " + str(time.time() - time_s) + " seconds")   

    # total_time = time.time() - start_time
    # time_s = time.time()
    # percent = nn_report(symbol, total_time, model, data, test_acc, valid_acc, 
    # train_acc, defaults["N_STEPS"], False)
    # y_real, y_pred = return_real_predict(model, data["X_valid"], data["y_valid"], data["column_scaler"][test_var], True)
    # print(f"real: {y_real}")
    # print(f"predict: {y_pred}")
    # predicted_price = predict(model, data, defaults["N_STEPS"], False) 
    # print("NN report took " + str(time.time() - time_s) + " seconds")

    # print(f"predicted value: {predicted_price}")

    # def load_nn_and_predict(symbol, current_date, params, model_dir, model_name):
    #     data, model = load_model_with_data(symbol, current_date, params, model_dir, model_name)

    #     # first grab the current price by getting the latest value from the og data frame
    #     y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var])
    #     real_y_values = y_real[-back_test_days:]
    #     current_price = real_y_values[-1]

    #     # then use predict fuction to get predicted price
    #     predicted_price = predict(model, data, params["N_STEPS"])

    
    
    


