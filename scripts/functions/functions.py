from xmlrpc.client import boolean
from config.environ import *
from functions.error import error_handler
import pandas as pd
import random
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
    if not os.path.isdir(f"{directory_dict['model']}/{save_folder}"):
        os.mkdir(f"{directory_dict['model']}/{save_folder}")
    if not os.path.isdir(f"{directory_dict['reports']}/{symbol}"):
        os.mkdir(f"{directory_dict['reports']}/{symbol}")

def check_prediction_subfolders(directory, nn_name):
    if not os.path.isdir(f"{directory}/{nn_name}"):
        os.mkdir(f"{directory}/{nn_name}")

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

def interpret_dict(dict):
    s = ""
    for i in dict:
        s += f"{i}={dict[i]}"
    return s

def n_max_elements(names, prices, N):
    final_list = []
  
    for i in range(0, N): 
        if len(prices) == 0:
            break

        max1 = 0          
        ind = 0
        for j in range(len(prices)):     
            if prices[j] > max1:
                max1 = prices[j]
                ind = j
        
        final_list.append(names[ind])
        prices.remove(prices[ind]);
        names.remove(names[ind])

          
    return final_list

def get_random_folder():
    return f"testing{random.randint(0, 99)}"
    

def get_model_name(nn_params):
    return (f"sh{'T' if nn_params['SHUFFLE'] else 'F'}"
            f"{features_string(nn_params['FEATURE_COLUMNS'])}{layers_string(nn_params['LAYERS'])}s"
            f"{nn_params['N_STEPS']}l{nn_params['LIMIT']}e{nn_params['EPOCHS']}"
            f"p{nn_params['PATIENCE']}b{nn_params['BATCH_SIZE']}d{nn_params['DROPOUT']}"
            f"t{nn_params['TEST_SIZE']}{nn_params['TEST_VAR']}")


def get_dtree_name(dt_params):
    return (f"{features_string(dt_params['FEATURE_COLUMNS'])}-md{dt_params['MAX_DEPTH']}-msl{dt_params['MIN_SAMP_LEAF']}")

def get_xtree_name(xt_params):
    return (f"{features_string(xt_params['FEATURE_COLUMNS'])}-md{xt_params['MAX_DEPTH']}-est{xt_params['N_ESTIMATORS']}"
        f"-msl{xt_params['MIN_SAMP_LEAF']}")

def get_rfore_name(rf_params):
    return (f"{features_string(rf_params['FEATURE_COLUMNS'])}-md{rf_params['MAX_DEPTH']}-est{rf_params['N_ESTIMATORS']}"
        f"-msl{rf_params['MIN_SAMP_LEAF']}")

def get_knn_name(knn_params):
    return f"{features_string(knn_params['FEATURE_COLUMNS'])}-nei{knn_params['N_NEIGHBORS']}-w{knn_params['WEIGHTS']}"

def get_ada_name(ada_params):
    return (f"{features_string(ada_params['FEATURE_COLUMNS'])}-md{ada_params['MAX_DEPTH']}-est{ada_params['N_ESTIMATORS']}"
        f"-msl{ada_params['MIN_SAMP_LEAF']}")

def get_xgb_name(xgb_params):
    return (f"{features_string(xgb_params['FEATURE_COLUMNS'])}-md{xgb_params['MAX_DEPTH']}-est{xgb_params['N_ESTIMATORS']}"
        f"-ml{xgb_params['MAX_LEAVES']}")

def get_bagreg_name(bag_params):
    return (f"{features_string(bag_params['FEATURE_COLUMNS'])}-md{bag_params['MAX_DEPTH']}-est{bag_params['N_ESTIMATORS']}"
        f"-msl{bag_params['MIN_SAMP_LEAF']}")

def get_mlens_name(mlens_params):
    return f"{features_string(mlens_params['FEATURE_COLUMNS'])}-l{mlens_params['LAYERS']}-m_est{mlens_params['META_EST']}"

def get_mlp_name(mlp_params):
    return (f"{features_string(mlp_params['FEATURE_COLUMNS'])}-l{mlp_params['LAYERS']}-stp{mlp_params['EARLY_STOP']}"
        f"-p{mlp_params['PATIENCE']}")

def get_test_name(params):
    test_name = str(params['ENSEMBLE'])
    for predictor in params['ENSEMBLE']:
        if "nn" in predictor:
            test_name += get_model_name(params[predictor])
        elif "DTREE" in predictor:
            test_name += get_dtree_name(params[predictor])
        elif "XTREE" in predictor:
            test_name += get_xtree_name(params[predictor])
        elif "RFORE" in predictor:
            test_name += get_rfore_name(params[predictor])
        elif "KNN" in predictor:
            test_name += get_knn_name(params[predictor])
        elif "ADA" in predictor:
            test_name += get_ada_name(params[predictor])
        elif "XGB" in predictor:
            test_name += get_xgb_name(params[predictor])
        elif "BAGREG" in predictor:
            test_name += get_bagreg_name(params[predictor])
        elif "MLENS" in predictor:
            test_name += get_mlens_name(params[predictor])
        elif "MLP" in predictor:
            test_name += get_mlp_name(params[predictor])
        else:
            test_name += "TEMP_FIX_NAME"

    if len(test_name) > 190:
        test_name = test_name[:190]
    return test_name

def features_string(features):
    string = "["
    for feature in features:
        string += f"""{feature.strip("'")},"""
    string = string[:-1]
    string += "]"

    return string

def layers_string(layers):
    string = "["
    for layer in layers:
        string += "(" + str(layer[0]) + "-" + layer_name_converter(layer) + ")"
    string += "]"

    return string

def layer_name_converter(layer):
    # print(layer, flush=True)
    string = ""
                     
    if str(layer[1]) == "<class 'keras.layers.rnn.lstm.LSTM'>":
        string += "LSTM"
    elif str(layer[1]) == "<class 'keras.layers.rnn.SimpleRNN'>":
        string += "SRNN"
    elif str(layer[1]) == "<class 'keras.layers.rnn.GRU'>":
        string += "GRU"
    elif str(layer[1]) == "<class 'keras.layers.core.dense.Dense'>":
        string += "Dense"
    elif str(layer[1]) == "<class 'tensorflow_addons.layers.esn.ESN'>":
        string += "ESN"
    else:
        string += str(layer[1])

    return string

def get_correct_direction(predicted_value, current_price, actual_price, test_var):
    if test_var == "acc":
        real_direction = int(boolean(actual_price > current_price))
        if real_direction == predicted_value:
            return 1.0
        else:
            return 0.0
    else:
        if ((predicted_value > current_price and actual_price > current_price) or 
        (predicted_value < current_price and actual_price < current_price)): 
            return 1.0
        elif predicted_value == current_price == actual_price:
            return 1.0
        elif predicted_value == current_price or actual_price == current_price: 
            return 0.5
        else:
            return 0.0

def percent_diff(pri1, pri2):
    return r2((abs(pri1 - pri2) / pri1) * 100)

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

def r0(num):
    return round(num, 0)

