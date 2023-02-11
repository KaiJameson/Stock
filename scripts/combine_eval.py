from config.symbols import sym_dict, combine_eval_dict, tune_year, tune_month, tune_day, tune_days
from config.model_repository import models
from functions.functions import check_directories,  get_test_name, r2
from functions.tuner import get_user_input
from paca_model import configure_gpu
from tuner import tuning
from make_excel import make_tuning_sheet
from sim_trades import simulate_trades
from itertools import combinations
import pandas as pd
import time
import copy
import sys



configure_gpu()

check_directories()

if len(sys.argv) > 3:
    s_time = time.perf_counter()

    sys.argv[2] = int(sys.argv[2])
    sys.argv[3] = int(sys.argv[3])

    if (type(sys.argv[2]) or type(sys.argv[3])) != type(0):
        print("Arguments 3 and 4 must be ints epressing the lower and upper bound on # of models to combine")
        print("Please try again")
        sys.exit(-1)

    
    results = []
    # print(combine_eval_dict)
    # print(type(combine_eval_dict))
    tune_symbols, do_not_use = get_user_input(sym_dict, combine_eval_dict)
    changable_dict = copy.deepcopy(combine_eval_dict)


    for comb_num in range(sys.argv[2], sys.argv[3] + 1):
        all_test_combinations = list(combinations(combine_eval_dict["ENSEMBLE"], comb_num))
        print(all_test_combinations)
        print(f"Now starting combination evaluation with params {combine_eval_dict}"
            f" with a total of {len(all_test_combinations)} combinations for groupsize {comb_num}", flush=True)

        for test_indexes in all_test_combinations:
            changable_dict["ENSEMBLE"] = []
            for index in test_indexes:
                changable_dict["ENSEMBLE"].append(index)
                changable_dict[index] = models[index]


            # print(f"\n\n{changable_dict}\n\n", flush=True)

            output_list = []
            equities = []
            for symbol in tune_symbols:
                # print(symbol)
                tuning_output = tuning(symbol, tune_year, tune_month, tune_day, tune_days, changable_dict, output=True)
                output_list.append(tuning_output)

            for method in combine_eval_dict["TRADE_METHOD"]:
                changable_dict["TRADE_METHOD"] = method
                equity = simulate_trades(tune_year, tune_month, tune_day, tune_days, changable_dict, output=True)
                equities.append(equity)

            # print(output_list)
            # print(equities)
            result_df = pd.DataFrame(output_list, columns=["Model Name", "Average percent", "Average direction", 
                "Time used", "Money Made"])
            result_df["Average percent"] = pd.to_numeric(result_df["Average percent"]).astype("float64")
            result_df["Average direction"] = pd.to_numeric(result_df["Average direction"]).astype("float64")
            test_output = [changable_dict["ENSEMBLE"], r2(result_df["Average percent"].mean()), r2(result_df["Average direction"].mean()),
                r2(result_df["Time used"].sum() / 60), r2(result_df["Money Made"].mean()), r2(equities[0]), r2(equities[1])]

            #result_df["Model Name"][0].split("]", 1)[0] + "]"

            # test_output.append(equities)
            print(f"\ntest output {test_output}")
            results.append(test_output)
            make_tuning_sheet(get_test_name(changable_dict), combine_eval_dict['TUNE_FOLDER'], tune_symbols)

    

    result_df = pd.DataFrame(results, columns=["Model Name", "Average percent", "Average direction", 
        "Time used", "Money Made", "Prepor No Rebal", "Rebal Split"])
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(result_df)

    result_df.to_csv(f"{combine_eval_dict['TUNE_FOLDER']}/summary/summary-{combine_eval_dict['ENSEMBLE']}.txt", sep="\t")
    print(f"Doing all of these tests took {(time.perf_counter() - s_time) / 60} minutes")
else:
    print("This program requires an argument for what symbols to run and two ints expressing # of models to combine.")
    print("Please try again")
    sys.exit(-1)