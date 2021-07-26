from environ import *
from error_functs import error_handler
import os


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
    return (str(params["FEATURE_COLUMNS"]) + "-limit" + str(params["LIMIT"]) + "-step" 
        + str(params["N_STEPS"]) + "-layer" + str(params["N_LAYERS"]) + "-unit" 
        + str(params["UNITS"]) + "-epoch" + str(params["EPOCHS"]) + "-pat" + str(params["PATIENCE"]) 
        + "-batch" + str(params["BATCH_SIZE"]) + "-drop" + str(params["DROPOUT"]))

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

def silence_tensorflow():
    import os
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # pass
