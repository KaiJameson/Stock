from time_functions import get_date_string, get_time_string, zero_pad_date_string
from environment import *
import urllib
import traceback
import time
import os
import sys


def check_directories():
    if not os.path.isdir(reports_directory):
        os.mkdir(reports_directory)
    if not os.path.isdir(config_directory):
        os.mkdir(config_directory)
    if not os.path.isdir(stock_decisions_directory):
        os.mkdir(stock_decisions_directory)
    if not os.path.isdir(graph_directory):
        os.mkdir(graph_directory)
    if not os.path.isdir(model_saveload_directory):
        os.mkdir(model_saveload_directory)
    if not os.path.isdir(tuning_directory):
        os.mkdir(tuning_directory)
    if not os.path.isdir(data_directory):
        os.mkdir(data_directory)
    if not os.path.isdir(excel_directory):
        os.mkdir(excel_directory)
    if not os.path.isdir(load_run_excel):
        os.mkdir(load_run_excel)
    if not os.path.isdir(current_price_directory):
        os.mkdir(current_price_directory)
    if not os.path.isdir(real_test_directory):
        os.mkdir(real_test_directory)
    if not os.path.isdir(results_directory):
       os.mkdir(results_directory)

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
    except:
        f = open(error_file, "a")
        f.write("Problem with deleting files in folder: " + directory + "\n")
        f.write(sys.exc_info()[1] + "\n")
        f.close()

def make_current_price(curr_price):
    date_string = get_date_string()

    f = open(current_price_directory + "/" + date_string + ".txt", "a")
    f.write(str(round(curr_price, 2)) + "\t")
    f.close()

def make_excel_file():
    date_string = get_date_string()

    fsym = open(excel_directory + "/" + date_string + "symbol" + ".txt", "r")
    sym_vals = fsym.read()
    fsym.close()

    freal = open(excel_directory + "/" + date_string + "real" + ".txt", "r")
    real_vals = freal.read()
    freal.close()

    fpred = open(excel_directory + "/" + date_string + "predict" + ".txt", "r")
    pred_vals = fpred.read()
    fpred.close()

    f = open(excel_directory + "/" + date_string + ".txt", "a+")
    
    f.write(sym_vals + "\n")
    f.write(str(real_vals) + "\n")
    f.write(str(pred_vals))
    f.close()

    os.remove(excel_directory + "/" + date_string + "symbol" + ".txt")
    os.remove(excel_directory + "/" + date_string + "real" + ".txt")
    os.remove(excel_directory + "/" + date_string + "predict" + ".txt")

def excel_output(symbol, real_price, predicted_price):
    date_string = get_date_string()

    f = open(excel_directory + "/" + date_string + "symbol" + ".txt", "a")
    f.write(symbol + ":" + "\t")
    f.close()

    f = open(excel_directory + "/" + date_string + "real" + ".txt", "a")
    f.write(str(round(real_price, 2)) + "\t")
    f.close()

    f = open(excel_directory + "/" + date_string + "predict" + ".txt", "a")
    f.write(str(round(predicted_price, 2)) + "\t")
    f.close()

def make_load_run_excel(symbol, train_acc, valid_acc, test_acc, from_real, percent_away):
    date_string = get_date_string()
    f = open(load_run_excel + "/" + date_string + ".txt", "a")
    f.write(symbol + "\t" + str(round(train_acc * 100, 2)) + "\t" + str(round(valid_acc * 100, 2)) + "\t" 
    + str(round(test_acc * 100, 2)) + "\t" + str(round(from_real, 2)) + "\t" + str(round(percent_away, 2)) 
    + "\n")
    f.close()

def real_test_excel(n_steps, lookup_step, test_size, n_layers, cell, units, dropout, bidirectional, loss, 
    optimizer, batch_size, epochs, patience, limit, feature_columns, avg_p, avg_d, avg_e, time_so_far, total_days):

    test_name = f"{feature_columns}-limit-{limit}-n_step-{n_steps}-layers-{n_layers}-units-{units}-epochs-{epochs}"
    f = open(real_test_directory + "/" + test_name + ".txt", "a")

    f.write("Parameters: n_steps: " + str(n_steps) + ", lookup step:" + str(lookup_step) + ", test size: " + str(test_size) + ",\n")
    f.write("N_layers: " + str(n_layers) + ", Cell: " + str(cell) + ",\n")
    f.write("Units: " + str(units) + "," + " Dropout: " + str(dropout) + ", Bidirectional: " + str(bidirectional) + ",\n")
    f.write("Loss: " + loss + ", Optimizer: " + optimizer + ", Batch_size: " + str(batch_size) + ",\n")
    f.write("Epochs: " + str(epochs) + ", Patience: " + str(patience) + ", Limit: " + str(limit) + ".\n")
    f.write("Feature Columns: " + str(feature_columns) + "\n\n")

    f.write("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent\n")
    f.write("and it predicted the correct direction " + avg_d + " percent of the time\n")
    f.write("while using an average of " + avg_e + " epochs.\n")
    f.write("Testing all of the days took " + str(round((time_so_far / 360), 0)) + " hours and " + str(round((time_so_far % 60), 2)) + " minutes.")
    f.close()

def error_handler(symbol, exception):
    f = open(error_file, "a")
    f.write("Problem encountered with stock: " + symbol + "\n")
    f.write("Error is of type: " + str(type(exception)))
    exit_info = sys.exc_info()
    f.write(str(exit_info[1]) + "\n")
    traceback.print_tb(tb=exit_info[2], file=f)
    f.close()
    print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

def interwebz_pls(symbol, end_date):
    if end_date != None:
        url = "https://api.polygon.io/v2/aggs/ticker/" + symbol + "/range/1/day/2000-01-01/" + str(end_date) + "?unadjusted=False&apiKey=AKVRXC2RTVB1C86IZI2M"
    else:
        time_now = zero_pad_date_string()
        url = "https://api.polygon.io/v2/aggs/ticker/" + symbol + "/range/1/day/2000-01-01/" + str(time_now) + "?unadjusted=False&apiKey=AKVRXC2RTVB1C86IZI2M"

    no_connection = True
    while no_connection:
        try:
            file = urllib.request.urlopen(url)
            print("connected")
            no_connection = False

        except urllib.error.URLError:
            print("no connect")
            time.sleep(1)