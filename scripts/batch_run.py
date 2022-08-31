from tuner import tuning
from config.symbols import tune_year, tune_month, tune_day, test_days, sym_dict
from config.model_repository import exhaustive_search
from config.environ import directory_dict
from functions.functions import check_directories, r2, get_test_name
from functions.tuner import get_user_input
from make_excel import make_tuning_sheet
from itertools import product
import pandas as pd
import sys
import copy
import time

check_directories()


if len(sys.argv) > 2:
    s_time = time.perf_counter()
    test_days = 500

    params = {
        "ENSEMBLE":[],
        "TRADING": False,
        "TUNE_FOLDER": directory_dict['batch_run'],
        "SAVE_FOLDER": "",
        "LIMIT": 4000,
    }

    print(f" ~~~ Starting rapid model testing now ~~~ ")

    params[sys.argv[2]] = copy.deepcopy(exhaustive_search[sys.argv[2]])
    params["ENSEMBLE"] = [sys.argv[2]]
    print(params)

    param_lists = []
    keys = []
    
    for variable in exhaustive_search[params["ENSEMBLE"][0]]:
        if type(exhaustive_search[params["ENSEMBLE"][0]][variable]) is list:
            
            params[params["ENSEMBLE"][0]][variable] = {}
            # print(params[params["ENSEMBLE"]][variable])
            keys.append(variable)
            param_lists.append(exhaustive_search[params["ENSEMBLE"][0]][variable])
            
    indexes = [len(i) for i in param_lists]
    j = 1
    for i in indexes:
        j *= i

    results = []
    print(f"\nIndexes have lengths of {indexes} with a total of {j} combinations\n\n")
    for index_tuple in product(*map(range, indexes)):
        for i, k in enumerate(indexes):
            params[params["ENSEMBLE"][0]][keys[i]] = param_lists[i][index_tuple[i]]

        # print(i)
        # print(keys[i])
        # print(param_lists[i][index_tuple[i]])
        print(params[params['ENSEMBLE'][0]])
        print(params[params['ENSEMBLE'][0]][keys[i]])

        tune_symbols, do_not_use = get_user_input(sym_dict, params[params['ENSEMBLE'][0]])
        # print(tune_symbols)
        # print(params[params['ENSEMBLE'][0]])
        print(f"Now starting test with adjustable params {params[params['ENSEMBLE'][0]]}")

        output_list = []
        for symbol in tune_symbols:
            # print(symbol)
            tuning_output = tuning(symbol, tune_year, tune_month, tune_day, test_days, params, output=True)
            output_list.append(tuning_output)

        print(output_list)
        result_df = pd.DataFrame(output_list, columns=["Model Name", "Average percent", "Average direction", 
            "Time used", "Money Made"])
        result_df["Average percent"] = pd.to_numeric(result_df["Average percent"]).astype("float64")
        result_df["Average direction"] = pd.to_numeric(result_df["Average direction"]).astype("float64")
        test_output = [result_df["Model Name"][0], r2(result_df["Average percent"].mean()), r2(result_df["Average direction"].mean()),
            r2(result_df["Time used"].sum() / 60), r2(result_df["Money Made"].mean())]

        results.append(test_output)
        make_tuning_sheet(get_test_name(params), params['TUNE_FOLDER'])
    

    result_df = pd.DataFrame(results, columns=["Model Name", "Average percent", "Average direction", 
        "Time used", "Money Made"])
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(result_df)

    result_df.to_csv(f"{params['TUNE_FOLDER']}/summary/exhaustive_{params['ENSEMBLE'][0]}.txt", sep="\t")
    print(f"Doing all of these tests took {(time.perf_counter() - s_time) / 60} minutes")

    
else:
    print("You must give this program two arguments in the style of \"sym#\"")
    print("then \"model abbreviation\" So that it knows tests to run and what symbols to use.")
    print("Please try again")
    sys.exit(-1)



    


