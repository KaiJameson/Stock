from time_functions import get_date_string, get_time_string, zero_pad_date_string
from environment import *
from symbols import trading_real_money
from api_key import real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key
import alpaca_trade_api as tradeapi
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

def check_model_subfolders(save_folder):
    if not os.path.isdir(model_saveload_directory + "/" + save_folder):
        os.mkdir(model_saveload_directory + "/" + save_folder)


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

def real_test_excel(test_year, test_month, test_day, n_steps, lookup_step, test_size, n_layers, cell, units, 
    dropout, bidirectional, loss, optimizer, batch_size, epochs, patience, limit, feature_columns, avg_p, 
    avg_d, avg_e, time_so_far, total_days):

    test_name = f"{feature_columns}-limit-{limit}-n_step-{n_steps}-layers-{n_layers}-units-{units}-epochs-{epochs}"
    f = open(real_test_directory + "/" + test_name + ".txt", "a")

    f.write("Parameters: N_steps: " + str(n_steps) + ", Lookup Step:" + str(lookup_step) + ", Test Size: " + str(test_size) + ",\n")
    f.write("N_layers: " + str(n_layers) + ", Cell: " + str(cell) + ",\n")
    f.write("Units: " + str(units) + "," + " Dropout: " + str(dropout) + ", Bidirectional: " + str(bidirectional) + ",\n")
    f.write("Loss: " + loss + ", Optimizer: " + optimizer + ", Batch_size: " + str(batch_size) + ",\n")
    f.write("Epochs: " + str(epochs) + ", Patience: " + str(patience) + ", Limit: " + str(limit) + ".\n")
    f.write("Feature Columns: " + str(feature_columns) + "\n\n")

    f.write("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent\n")
    f.write("and it predicted the correct direction " + avg_d + " percent of the time\n")
    f.write("while using an average of " + avg_e + " epochs.\n")
    f.write("The end day was: " + str(test_month) + "-" + str(test_day) + "-" + str(test_year) + ".\n")
    f.write("Testing all of the days took " + str((time_so_far // 3600)) + " hours and " + str(round((time_so_far % 60), 2)) + " minutes.")
    f.close()

def error_handler(symbol, exception):
    f = open(error_file, "a")
    f.write("Problem encountered with stock: " + symbol + "\n")
    f.write("Error is of type: " + str(type(exception)) + "\n")
    exit_info = sys.exc_info()
    f.write(str(exit_info[1]) + "\n")
    traceback.print_tb(tb=exit_info[2], file=f)
    f.close()
    print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

def interwebz_pls(symbol, end_date, conn_type):
    
    no_connection = True
    while no_connection:
        try:

            if trading_real_money:
                api = tradeapi.REST(real_api_key_id, real_api_secret_key, base_url="https://api.alpaca.markets")
            else:
                api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")

            if conn_type == "polygon":
                if end_date is not None:
                    df = api.polygon.historic_agg_v2(symbol, 1, "day", _from="2000-01-01", to=end_date).df
                else:
                    time_now = zero_pad_date_string()
                    df = api.polygon.historic_agg_v2(symbol, 1, "day", _from="2000-01-01", to=time_now).df

            if conn_type == "calendar":
                calendar = api.get_calendar(start=end_date, end=end_date)[0]

            no_connection = False

        except Exception:
            f = open(error_file, "a")
            f.write("\n\n EXCEPTION HANDLED \n\n")
            f.write("Error is of type: " + str(type(Exception)) + "\n")
            exit_info = sys.exc_info()
            f.write(str(exit_info[1]) + "\n")
            f.close()
            time.sleep(1)
            pass
            