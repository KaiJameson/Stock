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
from scipy.signal import cwt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def backtest_comparator(start_day, end_day, comparator, run_days):
    load_save_symbols = ["AGYS", "AMKR", "BG","BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
        "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]

    over_all = {i:[0.0, 0.0, 0.0]
    for i in range(start_day, end_day)}

    
    for symbol in load_save_symbols:
        print(symbol, flush=True)
        for i in range(start_day, end_day):
            data, train, valid, test = load_data(symbol, params=defaults["nn1"], end_date=None, shuffle=False, to_print=False)
            if comparator == "7MA":
                avg = MA_comparator(data, i, run_days)
            elif comparator == "lin_reg":
                avg = lin_reg_comparator(data, i, run_days)
            elif comparator == "EMA":
                avg = EMA_comparator(data, i, run_days)
            elif comparator == "TSF":
                avg = TSF_comparator(data, i, run_days)
            elif comparator == "sav_gol":
                if i == 1 or i == 3:
                    continue
                elif i % 2 == 0:
                    continue
                else:
                    avg = sav_gol_comparator(data, i, 4, run_days)


            over_all[i][0] += float(avg[0])
            over_all[i][1] += float(avg[1])
            over_all[i][2] += float(avg[2])

    print(f"~~~  {comparator}  ~~~")
    for j in range(start_day, end_day):
        print(f"{j}", end="")
        for metric in over_all[j]:
            print(f" {round(metric / len(load_save_symbols), 2)} ", end="")
        print()



if __name__ == "__main__":

    params = {
        "ENSEMBLE": ["nn1"],
        "TRADING": False,
        "SAVE_FOLDER": "tuning4",
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
            "EPOCHS": 10,
            "PATIENCE": 100,
            "SAVELOAD": True,
            "LIMIT": 4000,
            "FEATURE_COLUMNS": ["c"],
            "TEST_VAR": "c",
            "SAVE_PRED": {}
        }
    }
 
    # "macd", "macdsignal", "macdhist","balance_of_pow",  "parabolic_SAR_extended", "money_flow_ind", "7MA", "sc", "so", "sl", "sh", "sm", "sv"
 
    tuning(tune_year, tune_month, tune_day, tune_days, params)
    # ensemble_predictor("AGYS", params, get_current_datetime())

    year = 2021
    month = 5
    day = 15
    current_date = get_past_datetime(year, month, day)
    # print(f"year {year} month {month} day {day}")

    # df, blah, bal, alalal = load_data("AMKR", params["nn1"], current_date,  to_print=False)
    # df, train, valid, test = load_data("AGYS", params["nn1"], scale=False, shuffle=False, to_print=True)
    s = time.perf_counter()
    df2D = load_2D_data("AGYS", params["nn1"], end_date=current_date, shuffle=True, scale=True, to_print=False)
    # reg = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)
    reg = RandomForestRegressor(n_estimators=100)
    print(f"df2d took {time.perf_counter() - s}")
    s = time.perf_counter()
    reg.fit(df2D["X_train"], df2D["y_train"])
    print(f"fit took {time.perf_counter() - s}")
    print(f"""the last 250 days? {len(df2D["X_valid"][418:])}length of whole thing{len(df2D["X_valid"])}""")
    # print(f"help, depth{reg.get_depth()} leaves{reg.get_n_leaves()} params{reg.get_params()} ")
    print(f"""params{reg.get_params()}""")
    print(f"""score {reg.score(df2D["X_valid"], df2D["y_valid"])}""")

    # print(len(df("X_test")))

    # comparator_results_excel(df, 250, directory_dict["tuning"], "AGYS")
    # plot_graph(df["df"].c, df["df"].sc, "AGYS", 100, "c")
    # print(f"AGYS: {sav_gol_comparator(df, 7, 3, 3000)}", flush=True)
    # y_real, y_pred = return_real_predict()


    # backtest_comparator(5, 9, "sav_gol", 3000)
    # fuck_me_symbols = ["AGYS", "AMKR","BG", "BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
    #     "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]
    # for symbol in fuck_me_symbols:
    #     df, blah, bal, alalal = load_data(symbol, params["nn1"], to_print=False)
    #     print(symbol)
    #     print(f"{symbol}: {pre_c_comparator(df, 3000)}", flush=True)




        # TODO make sure to save the whole nn_params["SAVE_PRED"] dict at the end of tuner/backtester




