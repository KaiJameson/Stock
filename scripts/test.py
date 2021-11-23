import math
from config.environ import *
from config.symbols import *
from functions.tuner_functs import *
from functions.paca_model_functs import *
from functions.data_load_functs import *
from functions.io_functs import *
from functions.functions import *
from functions.time_functs import *
from paca_model import *
from load_run import *
from tuner import tuning
from statistics import mean


def backtest_comparator(start_day, end_day, comparator, run_days):
    load_save_symbols = ["AGYS", "AMKR", "BG","BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
        "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]

    over_all = {i:[0.0, 0.0, 0.0]
    for i in range(start_day, end_day)}

    
    for symbol in load_save_symbols:
        print(symbol, flush=True)
        for i in range(start_day, end_day):
            data, train, valid, test = load_data(symbol, defaults["nn1"], shuffle=False, to_print=False)
            if comparator == "7MA":
                avg = MA_comparator(data, i, run_days)
            elif comparator == "lin_reg":
                avg = lin_reg_comparator(data, i, run_days)
            elif comparator == "EMA":
                avg = EMA_comparator(data, i, run_days)
            elif comparator == "TSF":
                avg = TSF_comparator(data, i, run_days)
            elif comparator == "smooth_c":
                if i == 1 or i == 3:
                    continue
                elif i % 2 == 0:
                    continue
                else:
                    avg = smooth_c_comparator(data, i, 3, run_days)


            over_all[i][0] += float(avg[0])
            over_all[i][1] += float(avg[1])
            over_all[i][2] += float(avg[2])

    print(f"{comparator}")
    for j in range(start_day, end_day):
        print(f"{j}", end="")
        for metric in over_all[j]:
            print(f" {round(metric / len(load_save_symbols), 2)} ", end="")
        print()



if __name__ == "__main__":
    from config.symbols import *

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
    # model.load_weights(directory_dict["model"] + "/" + defaults["SAVE_FOLDER"] + "/" + model_name + ".h5")
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

    # def load_nn_and_predict(symbol, current_date, params, model, model_name):
    #     data, model = load_model_with_data(symbol, current_date, params, model, model_name)

    #     # first grab the current price by getting the latest value from the og data frame
    #     y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var])
    #     real_y_values = y_real[-back_test_days:]
    #     current_price = real_y_values[-1]

    #     # then use predict fuction to get predicted price
    #     predicted_price = predict(model, data, params["N_STEPS"])

    
    
    params = {
        "ENSEMBLE": ["sav_gol"],
        "TRADING": False,
        "SAVE_FOLDER": "tuning4",
        "TEST_VAR": "c",
        "nn1" : { 
            "N_STEPS": 100,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 0.2,
            "LAYERS": [(256, LSTM), (256, LSTM)],
            "UNITS": 256,
            "DROPOUT": .4,
            "BIDIRECTIONAL": False,
            "LOSS": "huber_loss",
            "OPTIMIZER": "adam",
            "BATCH_SIZE": 1024,
            "EPOCHS": 2000,
            "PATIENCE": 100,
            "SAVELOAD": True,
            "LIMIT": 4000,
            "FEATURE_COLUMNS": ["c"]
        }
    }

    # nn_train_save("AGYS", params=params)
    # load_trade(["AGYS"], params)

    # tuning(tune_year, tune_month, tune_day, tune_days, params)
    # ensemble_predictor("AGYS", params, get_current_datetime())
    # current_date = get_past_datetime(2020, 6, 1)

    # df, blah, bal, alalal = load_data("AGYS", params["nn1"], current_date, test_var="c", to_print=False)

    # comparator_results_excel(df, 250, directory_dict["tuning"], "AGYS")
    # plot_graph(df["df"].c, df["df"].sc, "AGYS", 100, "c")
    # print(f"AGYS: {MA_comparator(df, 7, 3000)}", flush=True)
    # y_real, y_pred = return_real_predict()

    backtest_comparator(2, 52, "TSF", 3000)

    # for symbol in load_save_symbols:
    #     df, blah, bal, alalal = load_data(symbol, params["nn1"], test_var="c", to_print=False)
    #     print(f"{symbol}: {smooth_c_comparator(df, 7, 3, 250)}", flush=True)




