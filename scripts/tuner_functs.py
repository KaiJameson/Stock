import sys
import copy

def grab_index(index_dict, params):
    while index_dict["n_step_in"] < len(params["N_STEPS"]):
        while index_dict["unit_in"] < len(params["UNITS"]):
            while index_dict["drop_in"] < len(params["DROPOUT"]):
                while index_dict["epochs_in"] < len(params["EPOCHS"]):
                    while index_dict["patience_in"] < len(params["PATIENCE"]):
                        while index_dict["limit_in"] < len(params["LIMIT"]):
                            index_dict["limit_in"] += 1
                            if index_dict["limit_in"] < len(params["LIMIT"]):
                                return index_dict
                        index_dict["patience_in"] += 1
                        index_dict["limit_in"] %= len(params["LIMIT"])
                        if index_dict["patience_in"] < len(params["PATIENCE"]):
                            return index_dict
                    index_dict["epochs_in"] += 1
                    index_dict["patience_in"] %= len(params["PATIENCE"])
                    if index_dict["epochs_in"] < len(params["EPOCHS"]):
                        return index_dict
                index_dict["drop_in"] += 1
                index_dict["epochs_in"] %= len(params["EPOCHS"])
                if index_dict["drop_in"] < len(params["DROPOUT"]):
                    return index_dict
            index_dict["unit_in"] += 1
            index_dict["drop_in"] %= len(params["DROPOUT"])
            if index_dict["unit_in"] < len(params["UNITS"]):
                return index_dict
        index_dict["n_step_in"] += 1
        index_dict["unit_in"] %= len(params["UNITS"])
        if index_dict["n_step_in"] < len(params["N_STEPS"]):
            return index_dict
    return index_dict

def change_params(index_dict, params):
    new_params = copy.deepcopy(params)
    new_params["N_STEPS"] =  params["N_STEPS"][index_dict["n_step_in"]]
    new_params["UNITS"] = params["UNITS"][index_dict["unit_in"]]
    new_params["DROPOUT"] = params["DROPOUT"][index_dict["drop_in"]]
    new_params["EPOCHS"] = params["EPOCHS"][index_dict["epochs_in"]]
    new_params["PATIENCE"] = params["PATIENCE"][index_dict["patience_in"]]
    new_params["LIMIT"] = params["LIMIT"][index_dict["limit_in"]]

    return new_params

def get_user_input(tune_sym_dict):
    if len(sys.argv) > 1:
        if sys.argv[1] == "tuning1":
            tune_symbols = tune_sym_dict[sys.argv[1]]
        elif sys.argv[1] == "tuning2":
            tune_symbols = tune_sym_dict[sys.argv[1]]
        elif sys.argv[1] == "tuning3":
            tune_symbols = tune_sym_dict[sys.argv[1]]
        elif sys.argv[1] == "tuning4":
            tune_symbols = tune_sym_dict[sys.argv[1]]
        elif sys.argv[1] == "tuning5":
            tune_symbols = tune_sym_dict[sys.argv[1]]
        else:
            print("You must give this program an argument in the style of \"tuning#\"")
            print("So that it knows what folder to save your models into.")
            print("Please try again")
            sys.exit(-1)

        master_params["SAVE_FOLDER"] = sys.argv[1]
        return tune_symbols

    else:
        print("You need to provide a second argument that says which tuning file ")
        print("and symbols you want to use. Please try again")
        sys.exit(-1)

