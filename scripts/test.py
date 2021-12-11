import math
from config.environ import *
from config.symbols import *
from config.api_key import *
from functions.tuner_functs import *
from functions.paca_model_functs import *
from functions.data_load_functs import *
from functions.io_functs import *
from functions.functions import *
from functions.time_functs import *
from functions.tech_functs import *
from paca_model import *
from load_run import *
from tuner import tuning
from statistics import mean
from scipy.signal import cwt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from tensorflow_addons.layers import ESN
from iexfinance.stocks import Stock, get_historical_data
import requests


def backtest_comparator(start_day, end_day, comparator, run_days):
    load_save_symbols = ["AGYS", "AMKR", "BG","BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
        "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]

    over_all = {i:[0.0, 0.0, 0.0]
    for i in range(start_day, end_day)}

    
    for symbol in load_save_symbols:
        print(symbol, flush=True)
        for i in range(start_day, end_day):
            data, train, valid, test = load_3D_data(symbol, params=defaults["nn1"], end_date=None, shuffle=False, to_print=False)
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
        "ENSEMBLE": ["KNN1"],
        "TRADING": False,
        "SAVE_FOLDER": "tune4",
        "nn1" : { 
            "N_STEPS": 100,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 0.2,
            "LAYERS": [(256, Dense), (256, Dense), (256, Dense), (256, Dense)],
            "DROPOUT": .4,
            "BIDIRECTIONAL": False,
            "LOSS": "huber_loss",
            "OPTIMIZER": "adam",
            "BATCH_SIZE": 1024,
            "EPOCHS": 2000,
            "PATIENCE": 200,
            "SAVELOAD": True,
            "LIMIT": 4000,
            "FEATURE_COLUMNS": ["c", "o"],
            "TEST_VAR": "c",
            "SAVE_PRED": {}
        },
        "DTREE1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v"],
            "MAX_DEPTH": 99,
            "MIN_SAMP_LEAF": 1,
            "LIMIT": 4000,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 0.2,
            "TEST_VAR": "c"
        },
        "RFORE1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v"],
            "MAX_DEPTH": 99,
            "MIN_SAMP_LEAF": 1,
            "LIMIT": 4000,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 0.2,
            "TEST_VAR": "c"
        },
        "KNN1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v"],
            "N_NEIGHBORS": 5,
            "LIMIT": 4000,
            "LOOKUP_STEP":1,
            "TEST_SIZE": 0.2,
            "TEST_VAR": "c"
        }
    }
 


    # start = datetime.datetime(2006, 12, 29)

    # end = datetime.datetime(2021, 12, 10)

    # os.environ["IEX_API_VERSION"] = "sandbox"

    # # aapl = Stock("AAPL", )
    # s = time.perf_counter()
    # # hello = get_historical_data("AAPL", start, end, output_format="pandas", token=iex_key)
    # hello = get_historical_data("AGYS", start, end, output_format="pandas", token=iex_sandbox_key)
    # print(f"it took {time.perf_counter() - s}")

    # print(hello)
    # print(len(hello))
    # print(type(hello))

    pd.set_option("display.max_columns", None)

    

    # symbol = "VIXY"
    # output_size = "full"
    # s = time.perf_counter()
    # url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={output_size}&apikey={alpha_key}"
    # r = requests.get(url)
    # data = r.json()
    # # print(data)
    # df = pd.DataFrame(data["Time Series (Daily)"])
    # print(f"alpha vantage bro {time.perf_counter() - s}")
    # s = time.perf_counter()
    # df = pd.DataFrame(data["Time Series (Daily)"])
    # df = df.transpose()
    # print(f"len {len(df)}")
    # print(df.head(100))
    
    # print(f" took {time.perf_counter() - s} to get dataframe in proper order")
    # # print(df_dict)
    
    # s = time.perf_counter()
    # read_saved_contents("../tmp.txt", load_dictionary)
    # df = pd.DataFrame(load_dictionary)
    # print(f"loading takes {time.perf_counter() - s}")
    
    # POTENTIAL STRUCTURE FOR DATA SAVING/LOADING CODE
    # check for a saved data file
    #   if saved file exists load it into dataframe
    #     
    # if not pull for current use and save it at the same time
    #  
    # check to see that all calendar testing days are in df
    #
    # def idk(symbol, output_size):
    #     load_dictionary = {
    #         "1. open": {},
    #         "2. high": {},
    #         "3. low": {},
    #         "4. close": {},
    #         "5. adjusted close": {},
    #         "6. volume": {},
    #         "7. dividend amount": {},
    #         "8. split coefficient": {}
    #     }
    #     if os.path.isfile(f"""{directory_dict["data"]}/{symbol}.txt"""):
    #         load_dictionary = read_saved_contents("""{directory_dict["data"]}/symbol.txt""", load_dictionary)
    #         df = pd.DataFrame(load_dictionary)
    #     else:
    #         df = get_alpha_dataframe(symbol, output_size)

    # features = ["sc", "so", "sl", "sh", "m", "sv", "up_band", "low_band", "OBV", "RSI", "lin_reg", "lin_reg_ang", "lin_reg_int", "lin_reg_slope", "pears_cor",
    #     "mon_flow_ind", "willR", "std_dev", "min", "max", "commod_chan_ind", "para_SAR", "para_SAR_ext", "rate_of_change", "ht_dcperiod", "ht_trendmode",
    #     "ht_dcphase", "ht_inphase", "quadrature", "ht_sine", "ht_leadsine", "ht_trendline", "mom", "abs_price_osc", "KAMA", "typ_price", "ult_osc", "chai_line",
    #     "chai_osc", "norm_avg_true_range", "median_price", "var", "aroon_down", "aroon_up", "aroon_osc", "bal_of_pow", "chande_mom_osc", "MACD", "MACD_signal",
    #     "MACD_hist", "con_MACD", "con_MACD_signal", "con_MACD_hist", "fix_MACD", "fix_MACD_signal", "fix_MACD_hist", "min_dir_ind", "min_dir_mov", "plus_dir_ind",
    #     "plus_dir_mov", "per_price_osc", "stoch_fast_k", "stoch_fast_d", "stoch_rel_stren_k", "stoch_rel_stren_d", "stoch_slowk", "stoch_slowd", "TRIX",
    #     "weigh_mov_avg", "DEMA", "EMA", "MESA_mama", "MESA_fama", "midpnt", "midprice", "triple_EMA", "tri_MA", "avg_dir_mov_ind", "true_range", "avg_price",
    #     "weig_c_price", "beta", "TSF", "day_of_week"]

    # def modify_dataframe(features, df):
    #     base_features = ["o", "c", "l", "h", "v"]
    #     for feature in features:
    #         if feature not in base_features:
    #             if techs_dict[feature]:
    #                 print(feature)
    #                 # df = techs_dict[feature]["function"](feature, df)
    #                 techs_dict[feature]["function"](feature, df)
    #             else:
    #                 print("Feature is not in the technical indicators dictionary. That sucks, probably")

    #     print(df)

    # df = get_alpaca_data("AGYS", None, get_api(), limit=4000)
    # s = time.perf_counter()
    # modify_dataframe(features, df)
    # print(f"modifications time!!! {time.perf_counter() - s}")


    # def get_alpha_dataframe(symbol, output_size):
    #     url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={output_size}&apikey={alpha_key}"
    #     r = requests.get(url)
    #     data = r.json()
    #     df = pd.DataFrame(data["Time Series (Daily)"])
    #     df = df.transpose()
    #     df_dict = df.to_dict()
    #     save_to_dictionary("../tmp.txt", df_dict)
        
    #     return df

    

    """
    LOAD ALL DATA
    make sure that each feature column is loaded 
    before loading new dataframe check that it doesn't already exit in dataframe dictionary
    by default return unmodified dataframe
    return 2D train/valid split data with a test that's never shuffled
    return 2D tensor if using Dense as first part of a model
    return 3D tensor if using recurrent models
    3d last sequence stuff will have to be dealt with somehow
    
    
    
    
    
    
    
    
    """

    # "macd", "macdsignal", "macdhist","balance_of_pow",  "parabolic_SAR_extended", "money_flow_ind", "7MA", "sc", "so", "sl", "sh", "sm", "sv"
 
    tuning(tune_year, tune_month, tune_day, 250, params)

    # year = 2021
    # month = 5
    # day = 15
    # current_date = get_past_datetime(year, month, day)
    # print(f"year {year} month {month} day {day}")

    # df, blah, bal, alalal = load_3D_data("AMKR", params["nn1"], current_date,  to_print=False)
    # df, train, valid, test = load_3D_data("AGYS", params["nn1"], scale=False, shuffle=False, to_print=True)

    # s = time.perf_counter()
    # df2D = load_2D_data("AGYS", params["KNN1"], end_date=current_date, shuffle=True, scale=True, to_print=False)
    # knn = KNeighborsRegressor(n_neighbors=5)
    # print(f"df2d took {time.perf_counter() - s}")
    # s = time.perf_counter()
    # knn.fit(df2D["X_train"], df2D["y_train"])
    # print(f"fit took {time.perf_counter() - s}")
    # print(f"""the last 250 days? {len(df2D["X_valid"][418:])}length of whole thing{len(df2D["X_valid"])}""")
    # print(f"""params{knn.get_params()}""")
    # print(f"""score {knn.score(df2D["X_valid"], df2D["y_valid"])}""")

    # print(len(df("X_test")))

    # comparator_results_excel(df, 250, directory_dict["tuning"], "AGYS")
    # plot_graph(df["df"].c, df["df"].sc, "AGYS", 100, "c")
    # print(f"AGYS: {sav_gol_comparator(df, 7, 3, 3000)}", flush=True)
    # y_real, y_pred = return_real_predict()


    # backtest_comparator(5, 9, "sav_gol", 3000)
    # fuck_me_symbols = ["AGYS", "AMKR","BG", "BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
    #     "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]
    # for symbol in fuck_me_symbols:
    #     df, blah, bal, alalal = load_3D_data(symbol, params["nn1"], to_print=False)
    #     print(symbol)
    #     print(f"{symbol}: {pre_c_comparator(df, 3000)}", flush=True)






