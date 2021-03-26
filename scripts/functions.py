from environ import *
from error_functs import error_handler
import traceback
import time
import os
import sys


def check_directories():
    for directory in directory_dict:
        if not os.path.isdir(directory_dict[directory]):
            os.mkdir(directory_dict[directory])

def check_model_folders(save_folder, symbol):
    if not os.path.isdir(directory_dict["model_directory"] + "/" + save_folder):
        os.mkdir(directory_dict["model_directory"] + "/" + save_folder)
    if not os.path.isdir(directory_dict["reports_directory"] + "/" + symbol):
        os.mkdir(directory_dict["reports_directory"] + "/" + symbol)

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
    return (str(params["FEATURE_COLUMNS"]) + "-limit-" + str(params["LIMIT"]) + "-n_step-" 
        + str(params["N_STEPS"]) + "-layers-" + str(params["N_LAYERS"]) + "-units-" 
        + str(params["UNITS"]) + "-epochs-" + str(params["EPOCHS"]))

def get_correct_direction(predicted_price, current_price, actual_price):
    if ((predicted_price > current_price and actual_price > current_price) or 
    (predicted_price < current_price and actual_price < current_price)): 
        correct_dir = 1.0
    elif predicted_price == current_price or actual_price == current_price: 
        correct_dir = 0.5
    else:
        correct_dir = 0.0

    return correct_dir

def silence_tensorflow():
    import os
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
