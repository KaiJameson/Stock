from config.environ import *
from functions.error_functs import error_handler
import pandas as pd
import os


def check_directories():
    for directory in directory_dict:
        sep = directory_dict[directory].split("/")
        while len(sep) > 2:
            sep = sep[:-1]
            current_dir = ("/".join(sep))
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)    
        if not os.path.isdir(directory_dict[directory]):
            os.mkdir(directory_dict[directory])

def check_model_folders(save_folder, symbol):
    if not os.path.isdir(directory_dict["model"] + "/" + save_folder):
        os.mkdir(directory_dict["model"] + "/" + save_folder)
    if not os.path.isdir(directory_dict["reports"] + "/" + symbol):
        os.mkdir(directory_dict["reports"] + "/" + symbol)

def check_prediction_subfolders(nn_name):
    if not os.path.isdir(f"""{directory_dict["save_predicts"]}/{nn_name}"""):
        os.mkdir(f"""{directory_dict["save_predicts"]}/{nn_name}""")

def delete_files(dirObject, dirPath):
    if dirObject.is_dir(follow_symlinks=False):
        name = os.fsdecode(dirObject.name)
        newDir = dirPath + "/" + name
        moreFiles = os.scandir(newDir)
        for f in moreFiles:
            if f.is_dir(follow_symlinks=False):
                delete_files(f, newDir)
                os.rmdir(newDir + "/" + os.fsdecode(f.name))
            else:
                os.remove(newDir + "/" + os.fsdecode(f.name))
        os.rmdir(newDir)
    else:
        os.remove(dirPath + "/" + os.fsdecode(dirObject.name))


def delete_files_in_folder(directory):
    try:
        files = os.scandir(directory)
        for f in files:
            delete_files(f, directory)
    except Exception:
        error_handler("Deleting files ", Exception)


def get_model_name(nn_params):
    return (f"""{nn_params["FEATURE_COLUMNS"]}-layers{layers_string(nn_params["LAYERS"])}-step"""
            f"""{nn_params["N_STEPS"]}-limit{nn_params["LIMIT"]}-epoch{nn_params["EPOCHS"]}""" 
            f"""-pat{nn_params["PATIENCE"]}-batch{nn_params["BATCH_SIZE"]}-drop{nn_params["DROPOUT"]}"""
            f"""-ts{nn_params["TEST_SIZE"]}{nn_params["TEST_VAR"]}""")

def get_dtree_name(dt_params):
    return (f"""{dt_params["FEATURE_COLUMNS"]}-md{dt_params["MAX_DEPTH"]}-msl{dt_params["MIN_SAMP_LEAF"]}""")

def get_rfore_name(rf_params):
    return (f"""{rf_params["FEATURE_COLUMNS"]}-md{rf_params["MAX_DEPTH"]}-est{rf_params["N_ESTIMATORS"]}"""
        f"""-msl{rf_params["MIN_SAMP_LEAF"]}""")

def get_knn_name(knn_params):
    return f"""{knn_params["FEATURE_COLUMNS"]}nei{knn_params["N_NEIGHBORS"]}"""

def get_ada_name(ada_params):
    return (f"""{ada_params["FEATURE_COLUMNS"]}-md{ada_params["MAX_DEPTH"]}-est{ada_params["N_ESTIMATORS"]}"""
        f"""-msl{ada_params["MIN_SAMP_LEAF"]}""")

def get_test_name(params):
    test_name = str(params["ENSEMBLE"])
    for predictor in params["ENSEMBLE"]:
        if "nn" in predictor:
            model_params = params[predictor]
            test_name += (predictor + str(model_params["FEATURE_COLUMNS"]) 
                + layers_string(model_params["LAYERS"]) + "s" + str(model_params["N_STEPS"]) 
                + "l" + str(model_params["LIMIT"]) + "e" + str(model_params["EPOCHS"]) 
                + "p" + str(model_params["PATIENCE"]) + "b" + str(model_params["BATCH_SIZE"]) 
                + "d" + str(model_params["DROPOUT"]) + "t" + str(model_params["TEST_SIZE"])
                + model_params["TEST_VAR"])
        elif "DTREE" in predictor:
            test_name += get_dtree_name(params[predictor])
        elif "RFORE" in predictor:
            test_name += get_rfore_name(params[predictor])
        elif "KNN" in predictor:
            test_name += get_knn_name(params[predictor])
        elif "ADA" in predictor:
            test_name += get_ada_name(params[predictor])

    if len(test_name) > 200:
        test_name = test_name[:200]
    return test_name

def layers_string(layers):
    string = "["
    for layer in layers:
        string += "(" + str(layer[0]) + "-" + layer_name_converter(layer) + ")"
    string += "]"

    return string

def layer_name_converter(layer):
    # print(layer, flush=True)
    string = ""
    
    if str(layer[1]) == "<class 'keras.layers.recurrent_v2.LSTM'>":
        string += "LSTM"
    elif str(layer[1]) == "<class 'keras.layers.recurrent.SimpleRNN'>":
        string += "SRNN"
    elif str(layer[1]) == "<class 'keras.layers.recurrent_v2.GRU'>":
        string += "GRU"
    elif str(layer[1]) == "<class 'keras.layers.core.dense.Dense'>":
        string += "Dense"
    elif str(layer[1]) == "<class 'tensorflow_addons.layers.esn.ESN'>":
        string += "ESN"
    else:
        string += str(layer[1])

    return string

def get_correct_direction(predicted_price, current_price, actual_price):
    if ((predicted_price > current_price and actual_price > current_price) or 
    (predicted_price < current_price and actual_price < current_price)): 
        return 1.0
    elif predicted_price == current_price == actual_price:
        return 1.0
    elif predicted_price == current_price or actual_price == current_price: 
        return 0.5
    else:
        return 0.0

def percent_from_real(y_real, y_predict):
    the_diffs = []
    for i in range(len(y_real) - 1):
        per_diff = (abs(y_real[i] - y_predict[i])/y_real[i]) * 100
        the_diffs.append(per_diff)
    pddf = pd.DataFrame(data=the_diffs)
    pddf = pddf.values
    return round(pddf.mean(), 2)

def sr1002(num):
        return str(round(num * 100, 2))

def sr2(num):
    return str(round(num, 2))

def r1002(num):
    return round(num * 100, 2)

def ra1002(num):
    return round(abs(num) * 100, 2)

def r2(num):
    return round(num, 2)
