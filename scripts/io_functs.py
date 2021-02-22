from environ import directory_dict
from time_functs import get_date_string
import os
import ast

def make_current_price(curr_price):
    date_string = get_date_string()

    f = open(directory_dict["current_price_directory"] + "/" + date_string + ".txt", "a")
    f.write(str(round(curr_price, 2)) + "\t")
    f.close()

def make_excel_file():
    date_string = get_date_string()

    fsym = open(directory_dict["excel_directory"] + "/" + date_string + "symbol" + ".txt", "r")
    sym_vals = fsym.read()
    fsym.close()

    freal = open(directory_dict["excel_directory"] + "/" + date_string + "real" + ".txt", "r")
    real_vals = freal.read()
    freal.close()

    fpred = open(directory_dict["excel_directory"] + "/" + date_string + "predict" + ".txt", "r")
    pred_vals = fpred.read()
    fpred.close()

    f = open(directory_dict["excel_directory"] + "/" + date_string + ".txt", "a+")
    
    f.write(sym_vals + "\n")
    f.write(str(real_vals) + "\n")
    f.write(str(pred_vals))
    f.close()

    os.remove(directory_dict["excel_directory"] + "/" + date_string + "symbol" + ".txt")
    os.remove(directory_dict["excel_directory"] + "/" + date_string + "real" + ".txt")
    os.remove(directory_dict["excel_directory"] + "/" + date_string + "predict" + ".txt")

def excel_output(symbol, real_price, predicted_price):
    date_string = get_date_string()

    f = open(directory_dict["excel_directory"] + "/" + date_string + "symbol" + ".txt", "a")
    f.write(symbol + ":" + "\t")
    f.close()

    f = open(directory_dict["excel_directory"] + "/" + date_string + "real" + ".txt", "a")
    f.write(str(round(real_price, 2)) + "\t")
    f.close()

    f = open(directory_dict["excel_directory"] + "/" + date_string + "predict" + ".txt", "a")
    f.write(str(round(predicted_price, 2)) + "\t")
    f.close()

def make_load_run_excel(symbol, train_acc, valid_acc, test_acc, from_real, percent_away):
    date_string = get_date_string()
    f = open(directory_dict["load_run_results"] + "/" + date_string + ".txt", "a")
    f.write(symbol + "\t" + str(round(train_acc * 100, 2)) + "\t" + str(round(valid_acc * 100, 2)) + "\t" 
    + str(round(test_acc * 100, 2)) + "\t" + str(round(from_real, 2)) + "\t" + str(round(percent_away, 2)) 
    + "\n")
    f.close()

def backtest_excel(directory, test_name, test_year, test_month, test_day, params, avg_p, 
    avg_d, avg_e, time_so_far, total_days):

    f = open(directory + "/" + test_name + ".txt", "a")

    f.write("Parameters: N_steps: " + str(params["N_STEPS"]) + ", Lookup Step:" + str(params["LOOKUP_STEP"]) + ", Test Size: " + str(params["TEST_SIZE"]) + ",\n")
    f.write("N_layers: " + str(params["N_LAYERS"]) + ", Cell: " + str(params["CELL"]) + ",\n")
    f.write("Units: " + str(params["UNITS"]) + "," + " Dropout: " + str(params["DROPOUT"]) + ", Bidirectional: " + str(params["BIDIRECTIONAL"]) + ",\n")
    f.write("Loss: " + params["LOSS"] + ", Optimizer: " + params["OPTIMIZER"] + ", Batch_size: " + str(params["BATCH_SIZE"]) + ",\n")
    f.write("Epochs: " + str(params["EPOCHS"]) + ", Patience: " + str(params["PATIENCE"]) + ", Limit: " + str(params["LIMIT"]) + ".\n")
    f.write("Feature Columns: " + str(params["FEATURE_COLUMNS"]) + "\n\n")

    f.write("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent\n")
    f.write("and it predicted the correct direction " + avg_d + " percent of the time\n")
    f.write("while using an average of " + avg_e + " epochs.\n")
    f.write("The end day was: " + str(test_month) + "-" + str(test_day) + "-" + str(test_year) + ".\n")
    f.write("Testing all of the days took " + str((time_so_far // 3600)) + " hours and " + str(round((time_so_far % 60), 2)) + " minutes.")
    f.close()

def print_backtest_results(params, total_days, avg_p, avg_d, avg_e, year, month, day, time_so_far):
    print("Parameters: N_steps: " + str(params["N_STEPS"]) + ", Lookup Step:" + str(params["LOOKUP_STEP"]) + ", Test Size: " + str(params["TEST_SIZE"]) + ",")
    print("N_layers: " + str(params["N_LAYERS"]) + ", Cell: " + str(params["CELL"]) + ",")
    print("Units: " + str(params["UNITS"]) + "," + " Dropout: " + str(params["DROPOUT"]) + ", Bidirectional: " + str(params["BIDIRECTIONAL"]) + ",")
    print("Loss: " + params["LOSS"] + ", Optimizer: " + 
    params["OPTIMIZER"] + ", Batch_size: " + str(params["BATCH_SIZE"]) + ",")
    print("Epochs: " + str(params["EPOCHS"]) + ", Patience: " + str(params["PATIENCE"]) + ", Limit: " + str(params["LIMIT"]) + ".")
    print("Feature Columns: " + str(params["FEATURE_COLUMNS"]) + "\n\n")

    print("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent")
    print("and it predicted the correct direction " + avg_d + " percent of the time ")
    print("while using an average of " + avg_e + " epochs.")
    print("The end day was: " + str(month) + "-" + str(day) + "-" + str(year))
    print("Testing all of the days took " + str(time_so_far // 3600) + " hours and " + str(round((time_so_far % 60), 2)) + " minutes.")

def read_saved_contents(file_path, return_dict):
    f = open(file_path, "r")

    file_contents = {}
    for line in f:
        parts = line.strip().split(":")
        file_contents[parts[0]] = parts[1]
    f.close()

    for key in file_contents:
        print(str(key) + " " + str(file_contents[key]))
        if type(return_dict[key]) == type("str"):
            return_dict[key] = file_contents[key]
        elif type(return_dict[key]) == type(0):
            return_dict[key] = int(file_contents[key])
        elif type(return_dict[key]) == type(0.0):
            return_dict[key] = float(file_contents[key])
        elif type(return_dict[key]) == type([]):
            return_dict[key] = ast.literal_eval(file_contents[key])
        else:
            print("Unexpected type found in this file")
    
    return return_dict

def save_to_dictionary(file_path, dictionary):
    f = open(file_path, "w")

    for key in dictionary:
        f.write(str(key) + ":" + str(dictionary[key]) + "\n")

    f.close()

