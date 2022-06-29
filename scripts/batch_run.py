from tuner import tuning
from config.symbols import tune_year, tune_month, tune_day, test_days
from config.model_repository import exhaustive_search
from functions.functions import check_directories
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

        print(f"""Now starting test with adjustable params {params[params["ENSEMBLE"][0]]}""")
        test_output = tuning(tune_year, tune_month, tune_day, test_days, params, output=True)
        results.append(test_output)

    result_df = pd.DataFrame(results, columns=["Model Name", "Average percent", "Average direction", 
        "Time used", "Money Made"])
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(result_df)
    print(f"Doing all of these tests took {(time.perf_counter() - s_time)} minutes")

    
else:
    print("You must give this program two arguments in the style of \"tune#\"")
    print("then \"model abbreviation\" So that it knows tests to run and what symbols to use.")
    print("Please try again")
    sys.exit(-1)



    


