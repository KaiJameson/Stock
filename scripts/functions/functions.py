from config.environ import *
from functions.error_functs import error_handler
import pandas as pd
import os


def check_directories():
    for directory in directory_dict:
        if not os.path.isdir(directory_dict[directory]):
            os.mkdir(directory_dict[directory])

def check_model_folders(save_folder, symbol):
    if not os.path.isdir(directory_dict["model_dir"] + "/" + save_folder):
        os.mkdir(directory_dict["model_dir"] + "/" + save_folder)
    if not os.path.isdir(directory_dict["reports_dir"] + "/" + symbol):
        os.mkdir(directory_dict["reports_dir"] + "/" + symbol)

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


def get_test_name(params):
    return (str(params["FEATURE_COLUMNS"]) + "-layers" + layers_string(params["LAYERS"]) + "-step" 
        + str(params["N_STEPS"]) + "-limit" + str(params["LIMIT"]) + "-epoch" + str(params["EPOCHS"]) 
        + "-pat" + str(params["PATIENCE"]) + "-batch" + str(params["BATCH_SIZE"]) 
        + "-drop" + str(params["DROPOUT"]) + "-ts" + str(params["TEST_SIZE"]))

def layers_string(layers):
    string = "["
    for layer in layers:
        string += "(" + str(layer[0]) + "-" + layer_name_converter(layer) + ")"
    string += "]"

    return string

def layer_name_converter(layer):
    string = ""
    if str(layer[1]) == "<class 'tensorflow.python.keras.layers.recurrent_v2.LSTM'>":
        string += "LSTM"
    elif str(layer[1]) == "<class 'tensorflow.python.keras.layers.recurrent.SimpleRNN'>":
        string += "SRNN"
    elif str(layer[1]) == "<class 'tensorflow.python.keras.layers.recurrent_v2.GRU'>":
        string += "GRU"
    elif str(layer[1]) == "<class 'tensorflow.python.keras.layers.core.Dense'>":
        string += "Dense"
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

def silence_tensorflow():
    import os
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
