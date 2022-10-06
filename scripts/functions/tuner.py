from functions.data_load import df_subset, load_all_data
from functions.functions import get_random_folder
from functions.time import get_current_datetime
from paca_model import ensemble_predictor
import sys
import copy

def subset_and_predict(symbol, params, current_date, df_dict, to_print=True):
    sub_df = df_subset(df_dict, current_date)
    data_dict = load_all_data(params, sub_df, "one_day", to_print)
    
    predicted_price, current_price, epochs_run = ensemble_predictor(symbol, params, current_date, 
        data_dict, sub_df['price'])
    
    return predicted_price, current_price, epochs_run


def get_user_input(sym_dict, params):
    if len(sys.argv) > 1:
        if sys.argv[1] in sym_dict:
            tune_symbols = sym_dict[sys.argv[1]]
        else:
            print("You must give this program an argument in the style of \"sym#\"")
            print("So that it knows what symbols to use.")
            print("Please try again")
            sys.exit(-1)

        params["SAVE_FOLDER"] = get_random_folder()
        return tune_symbols, params

    else:
        print("You need to provide a second argument that says which tuning file ")
        print("and symbols you want to use. Please try again")
        sys.exit(-1)

