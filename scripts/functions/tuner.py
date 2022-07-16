from functions.time import increment_calendar
from functions.data_load import df_subset, load_all_data
from paca_model import ensemble_predictor
import sys
import copy

def subset_and_predict(symbol, params, current_date, master_df, to_print=True):
    sub_df = df_subset(current_date, master_df)
    data_dict = load_all_data(params, sub_df, to_print)

    predicted_price, current_price, epochs_run = ensemble_predictor(symbol, params, current_date, 
                    data_dict, sub_df)
    
    return predicted_price, current_price, epochs_run, data_dict, sub_df


def change_params(index_dict, params):
    new_params = copy.deepcopy(params)
    new_params["N_STEPS"] =  params["N_STEPS"][index_dict["n_step_in"]]
    new_params["UNITS"] = params["UNITS"][index_dict["unit_in"]]
    new_params["DROPOUT"] = params["DROPOUT"][index_dict["drop_in"]]
    new_params["EPOCHS"] = params["EPOCHS"][index_dict["epochs_in"]]
    new_params["PATIENCE"] = params["PATIENCE"][index_dict["patience_in"]]
    new_params["LIMIT"] = params["LIMIT"][index_dict["limit_in"]]

    return new_params

def get_user_input(tune_sym_dict, params):
    if len(sys.argv) > 1:
        if sys.argv[1] in tune_sym_dict:
            tune_symbols = tune_sym_dict[sys.argv[1]]
        else:
            print("You must give this program an argument in the style of \"tune#\"")
            print("So that it knows what folder to save your models into.")
            print("Please try again")
            sys.exit(-1)

        params["SAVE_FOLDER"] = sys.argv[1]
        return tune_symbols, params

    else:
        print("You need to provide a second argument that says which tuning file ")
        print("and symbols you want to use. Please try again")
        sys.exit(-1)




