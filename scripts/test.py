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
from sklearn.model_selection import KFold
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
        "ENSEMBLE": ["DTREE1"],
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


    params = {
        # "ENSEMBLE": ["nn1", "nn2"],
        # "ENSEMBLE": ["ADA1", "KNN1", "RFORE1"],
        "ENSEMBLE": ["nn1"],
        "TRADING": False,
        "SAVE_FOLDER": "tune4",
        "nn1" : { 
            "N_STEPS": 100,
            "LOOKUP_STEP": 1,
            "TEST_SIZE": 0.2,
            "LAYERS": [(256, LSTM), (256, Dense), (128, Dense), (64, Dense)],
            "UNITS": 256,
            "DROPOUT": .4,
            "BIDIRECTIONAL": False,
            "LOSS": "huber_loss",
            "OPTIMIZER": "adam",
            "BATCH_SIZE": 1024,
            "EPOCHS": 2000,
            "PATIENCE": 200,
            "LIMIT": 4000,
            "FEATURE_COLUMNS": ["o", "l", "h", "c", "m", "v", "tc", "vwap"],
            "TEST_VAR": "c",
            "SAVE_PRED": {}
            },
        "LIMIT": 4000,
    }

    symbol = "AGYS"
    predictor = "nn1"
    scale = True
    to_print = True


    df = get_proper_df(symbol, params[predictor]["LIMIT"], "V2")
    data_dict = load_all_data(params, df)


    tt_df, result = preprocess_dfresult(params[predictor], df, scale=scale, to_print=to_print)
    
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(tt_df[params[predictor]["FEATURE_COLUMNS"]].tail(params[predictor]["LOOKUP_STEP"]))
    # drop NaNs
    tt_df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=params[predictor]["N_STEPS"])
    for entry, target in zip(tt_df[params[predictor]["FEATURE_COLUMNS"]].values, tt_df["future"].values):
        sequences.append(entry)
        if len(sequences) == params[predictor]["N_STEPS"]:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result["last_sequence"] = last_sequence
    # print(last_sequence)
    # construct the X"s and y"s
    X, y = [], []
    
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    print(len(X), len(y))

    num_splits = 5
    
    accuracies = []
    # kfold = KFold(n_splits=num_splits, shuffle=True)
    kfold = KFold(n_splits=num_splits, shuffle=False)
    i = 0
    for train, test in kfold.split(X, y):
        print(f" IIIIIIIIIIIIIIII {i} \n\n ")
        print(f"len of train {len(X[train])}")
        print(f"len of test {len(y[test])}")
        print(f"what we're selecting {test}")
        i += 1

        train = Dataset.from_tensor_slices((X[train], y[train]))
        valid = Dataset.from_tensor_slices((X[test], y[test]))

        train = train.batch(params[predictor]["BATCH_SIZE"])
        valid = valid.batch(params[predictor]["BATCH_SIZE"])

        train = train.cache()
        valid = valid.cache()

        train = train.prefetch(buffer_size=AUTOTUNE)
        valid = valid.prefetch(buffer_size=AUTOTUNE)

        data_dict["train"] = train
        data_dict["valid"] = valid

        nn_params = params[predictor]
        
        check_model_folders(params["SAVE_FOLDER"], symbol)
    
        model_name = (symbol + "-" + get_model_name(nn_params))

    
        model = create_model(nn_params)

        logs_dir = "logs/" + get_time_string() + "-" + params["SAVE_FOLDER"]

        checkpointer = ModelCheckpoint(directory_dict["model"] + "/" + params["SAVE_FOLDER"] + "/" 
            + model_name + ".h5", save_weights_only=True, save_best_only=True, verbose=1)
        
        if save_logs:
            tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch="200, 1200") 
        else:
            tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch=0)

        # early_stop = EarlyStopping(patience=nn_params["PATIENCE"])
        
        history = model.fit(data_dict["train"],
            batch_size=nn_params["BATCH_SIZE"],
            epochs=nn_params["EPOCHS"],
            verbose=0,
            # validation_data=data_dict["valid"],
            callbacks = [tboard_callback, checkpointer]   
        )

        print(result["column_scaler"])
        y_real, y_pred = return_real_predict(model, X[test], y[test], result["column_scaler"]["c"])

        # y_real = y[test]
        # y_pred = model.predict(X[test])
        acc = get_accuracy(y_pred, y_real, lookup_step=1)
        print(r1002(acc))
        accuracies.append(acc)
        model.evaluate(valid)

        epochs_used = len(history.history["loss"])
            
        if not save_logs:
            delete_files_in_folder(logs_dir)
            os.rmdir(logs_dir)
    
    overall_acc = r1002(sum(accuracies) / num_splits)
    print(overall_acc)

