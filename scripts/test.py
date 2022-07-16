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
from functions.voltil_functs import *
from paca_model import *
from load_run import *
from tuner import tuning
from statistics import mean
from scipy.signal import cwt
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow_addons.layers import ESN
from iexfinance.stocks import Stock, get_historical_data
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from mlens.ensemble import SuperLearner
import math
import requests

import copy


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
        "ENSEMBLE": ["ADA1", "RFORE1", "XGB1", "nn1"],
        "TRADING": False,
        "SAVE_FOLDER": "tune4",
        "nn1" : { 
            "N_STEPS": 100,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 0.2,
            "LAYERS": [(256, Dense), (256, Dense), (256, Dense), (256, Dense)],
            "SHUFFLE": True,
            "DROPOUT": .4,
            "BIDIRECTIONAL": False,
            "LOSS": "huber_loss",
            "OPTIMIZER": "adam",
            "BATCH_SIZE": 1024,
            "EPOCHS": 2000,
            "PATIENCE": 200,
            "SAVELOAD": True,
            "LIMIT": 4000,
            "FEATURE_COLUMNS": ["c", "o", "l", "h", "m", "v"],
            "TEST_VAR": "c",
            "SAVE_PRED": {}
        },
        "DTREE1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v"],
            "MAX_DEPTH": 5,
            "MIN_SAMP_LEAF": 1,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "RFORE1" : {
            "FEATURE_COLUMNS": ["c", "vwap"],
            "N_ESTIMATORS": 1000,
            "MAX_DEPTH": 10000,
            "MIN_SAMP_LEAF": 1,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "KNN1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v", "tc", "vwap"],
            "N_NEIGHBORS": 5,
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "ADA1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v", "tc", "vwap"],
            "N_ESTIMATORS": 100,
            "MAX_DEPTH": 10000,
            "MIN_SAMP_LEAF": 1,
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "XGB1" : {
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v", "tc", "vwap"],
            "N_ESTIMATORS": 100,
            "MAX_DEPTH": 1000,
            "MAX_LEAVES": 1000,
            "GAMMA": 0.0,
            "LOOKUP_STEP":1,
            "TEST_SIZE": 1,
            "TEST_VAR": "c"
        },
        "LIMIT": 4000,
    }
 


    api = get_api()

    s = time.perf_counter()
    # news = api.get_news("QCOM", limit=5000)

    # print(news)
    # print(news[0])
    # for ele in news:
    #     print(ele.author, ele.created_at)
    # print(len(news))
    predictor = "RFORE1"

    df = get_proper_df("AGYS", 4000, "V2")
    data_dict = load_all_data(params, df)
    print(data_dict.keys())

    ensemble = SuperLearner(scorer=mean_squared_error, random_state=42)

    ensemble.add([RandomForestRegressor(random_state=42), LinearSVR(loss="squared_epsilon_insensitive", dual=False)])

    ensemble.add_meta(LinearRegression())

    ensemble.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    fore_pred = ensemble.predict(data_dict[predictor]["X_test"])
    scale = data_dict[predictor]["column_scaler"]["future"]
    fore_pred = np.array(fore_pred)
    fore_pred = fore_pred.reshape(1, -1)
    predicted_price = np.float32(scale.inverse_transform(fore_pred)[-1][-1])

    print(predicted_price)

    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # print(df)
    # print(data_dict["nn1"])
    
    


    print(time.perf_counter() - s, flush=True)

    #best so far for volitility 
    # symbol = "UVXY"
    
    # print(get_proper_df(symbol, 4000, "V2"))

    all_features = ["o", "l", "h", "c", "m", "v", "sc", "so", "sl", "sh", "sm", "sv", "7MA", "up_band", "low_band", "OBV", "RSI", "lin_reg", "lin_reg_ang", 
        "lin_reg_int", "lin_reg_slope", "pears_cor", "mon_flow_ind", "willR", "std_dev", "min", "max", "commod_chan_ind", "para_SAR", "para_SAR_ext", "rate_of_change", 
        "ht_dcperiod", "ht_trendmode", "ht_dcphase", "ht_inphase", "quadrature", "ht_sine", "ht_leadsine", "ht_trendline", "mom", "abs_price_osc", "KAMA", "typ_price", 
        "ult_osc", "chai_line", "chai_osc", "norm_avg_true_range", "median_price", "var", "aroon_down", "aroon_up", "aroon_osc", "bal_of_pow", "chande_mom_osc", "MACD", 
        "MACD_signal", "MACD_hist", "con_MACD", "con_MACD_signal", "con_MACD_hist", "fix_MACD", "fix_MACD_signal", "fix_MACD_hist", "min_dir_ind", "min_dir_mov", "plus_dir_ind",
        "plus_dir_mov", "per_price_osc", "stoch_fast_k", "stoch_fast_d", "stoch_rel_stren_k", "stoch_rel_stren_d", "stoch_slowk", "stoch_slowd", "TRIX",
        "weigh_mov_avg", "DEMA", "EMA", "MESA_mama", "MESA_fama", "midpnt", "midprice", "triple_EMA", "tri_MA", "avg_dir_mov_ind", "true_range", "avg_price",
        "weig_c_price", "beta", "TSF", "day_of_week"]

    direct_value_features = ["o", "l", "h", "c", "m", "sc", "so", "sl", "sh", "sm", "7MA", "up_band", "low_band", "lin_reg", "lin_reg_ang", "lin_reg_int", "lin_reg_slope", 
                "min", "max", "ht_trendline",  "KAMA", "typ_price", "median_price", "var", "TRIX", "weigh_mov_avg", "DEMA", "EMA", "MESA_mama", "MESA_fama", 
                "midpnt", "midprice", "triple_EMA", "tri_MA", "avg_price", "weig_c_price", "TSF"]

    

    def store_sci_predictors(params, data_dict, saved_models):
        for predictor in params["ENSEMBLE"]:
            if "DTREE" in predictor:
                tree = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
                    min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"])
                tree.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])     
                saved_models[predictor] = tree         
            # elif "RFORE" in predictor:

            # elif "KNN" in predictor:

        return saved_models


    # year = 2019
    # month = 12
    # day = 1
    # how_damn_long_to_run_for = 500
 
    # year = 2020
    # month = 5
    # day = 17
    # how_damn_long_to_run_for = 250

    # tuning(year, month, day, how_damn_long_to_run_for, params)

    # backtest_comparator(5, 9, "sav_gol", 3000)
    # fuck_me_symbols = ["AGYS", "AMKR","BG", "BGS", "CAKE", "CCJ", "DFS", "ELY", "FLEX", 
    #     "INTC", "JBLU", "LLNW", "NWL", "QCOM", "RDN", "SHO", "SMED", "STLD", "WERN", "ZION"]
    # for symbol in fuck_me_symbols:
    #     df, blah, bal, alalal = load_3D_data(symbol, params["nn1"], to_print=False)
    #     print(symbol)
    #     print(f"{symbol}: {pre_c_comparator(df, 3000)}", flush=True)


    

