from matplotlib import pyplot as plt
from config.environ import test_money, directory_dict
from functions.functions import percent_from_real, layers_string, get_model_name
from functions.time_functs import get_current_date_string, get_time_string
from functions.tuner_functs import MA_comparator, lin_reg_comparator, smooth_c_comparator
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

def write_nn_report(symbol, report_dir, total_minutes, real_y_values, predicted_y_values,
        curr_price, future_price, test_acc, valid_acc, train_acc, y_real, y_pred, test_var="c"):
    spencer_money = test_money * (curr_price/real_y_values[0])
    f = open(report_dir, "a")
    f.write("~~~~~~~" + symbol + "~~~~~~~\n")
    f.write("Spencer wants to have: $" + str(round(spencer_money, 2)) + "\n")
    money_made = model_money(test_money, real_y_values, predicted_y_values)
    f.write("Money made from using real vs predicted: $" + str(round(money_made, 2)) + "\n")
    per_mon = perfect_money(test_money, real_y_values)
    f.write("Money made from being perfect: $" + str(round(per_mon, 2)) + "\n")
    f.write("The test var was " + test_var + "\n")
    f.write("Total run time was: " + str(round(total_minutes, 2)) + " minutes.\n")
    f.write("The price at run time was: " + str(round(curr_price, 2)) + "\n")
    f.write("The predicted price for tomorrow is: " + str(future_price) + "\n")
    
    percent = future_price / curr_price
    if curr_price < future_price:
        f.write("That would mean a growth of: " + str(round((percent - 1) * 100, 2)) + "%\n")
        f.write("I would buy this stock.\n")
    elif curr_price > future_price:
        f.write("That would mean a loss of: " + str(abs(round((percent - 1) * 100, 2))) + "%\n")
        f.write("I would sell this stock.\n")
    
    f.write("The average away from the real is: " + str(percent_from_real(y_real, y_pred)) + "%\n")
    f.write("Test accuracy score: " + str(round(test_acc * 100, 2)) + "%\n")
    f.write("Validation accuracy score: " + str(round(valid_acc * 100, 2)) + "%\n")
    f.write("Training accuracy score: " + str(round(train_acc * 100, 2)) + "%\n")
    f.close()


def model_money(money, data1, data2):
    stocks_owned = 0
    for i in range(0 , len(data1) - 1):
        now_price = data1[i]
        predict_price = data2[i + 1]
        if predict_price > now_price:
            stocks_can_buy = money // now_price
            if stocks_can_buy > 0:
                money -= stocks_can_buy * now_price
                stocks_owned += stocks_can_buy
        elif predict_price < now_price:
            money += now_price * stocks_owned
            stocks_owned = 0
    if stocks_owned != 0:
        money += stocks_owned * data1[len(data1)-1]
    return money

def perfect_money(money, data):
    stonks_owned = 0
    for i in range(0, len(data) - 1):
        now_price = data[i]
        tommorow_price = data[i + 1]
        if tommorow_price > now_price:
            stonks_can_buy = money // now_price
            if stonks_can_buy > 0:
                money -= stonks_can_buy * now_price
                stonks_owned += stonks_can_buy
        elif tommorow_price < now_price:
            money += now_price * stonks_owned
            stonks_owned = 0
    if stonks_owned != 0:
        money += stonks_owned * data[len(data) - 1]
    return money

def make_runtime_price(curr_price):
    date_string = get_current_date_string()

    f = open(directory_dict["runtime_price"] + "/" + date_string + ".txt", "a")
    f.write(str(round(curr_price, 2)) + "\t")
    f.close()

def runtime_predict_excel(symbols, pred_curr_list):
    date_string = get_current_date_string()
    
    run_pre_text = ""
    for symbol in symbols:
        run_pre_text += f"{symbol}:\t"
    run_pre_text += "\n"

    for symbol in symbols:   
        run_pre_text += str(round(pred_curr_list[symbol]["predicted"], 2)) + "\t"
    run_pre_text += "\n"

    for symbol in symbols:
        run_pre_text += str(round(pred_curr_list[symbol]["current"], 2)) + "\t"

    f = open(directory_dict["runtime_predict"] + "/" + date_string + ".txt", "a+")
    f.write(run_pre_text)
    f.close()

def make_load_run_excel(symbol, train_acc, valid_acc, test_acc, from_real, percent_away):
    date_string = get_current_date_string()
    f = open(directory_dict["load_run_results"] + "/" + date_string + ".txt", "a")
    f.write(symbol + "\t" + str(round(train_acc * 100, 2)) + "\t" + str(round(valid_acc * 100, 2)) + "\t" 
    + str(round(test_acc * 100, 2)) + "\t" + str(round(from_real, 2)) + "\t" + str(round(percent_away, 2)) 
    + "\n")
    f.close()

def backtest_excel(directory, test_name, test_year, test_month, test_day, params, avg_p, 
    avg_d, avg_e, time_so_far, total_days, current_money, hold_money):

    file = open(directory + "/" + test_name + ".txt", "a")

    file.write("Testing finished for ensemble: " + str(params["ENSEMBLE"]) + "\n")
    file.write("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent\n")
    file.write("and it predicted the correct direction " + avg_d + " percent of the time\n")
    overall_epochs = []
    for predictor in params["ENSEMBLE"]:
        if "nn" in predictor:
            overall_epochs.append(avg_e[predictor])
    all_epochs = mean(overall_epochs)
    file.write(f"The models (if any) used {all_epochs}.")

    
    if current_money != None:
        file.write("If it was trading for real it would have made " + str(current_money) + " as compared to " + str(hold_money) + " if you held it.\n")
    file.write("Testing all of the days took " + str(round(time_so_far / 3600, 2)) + " hours or " + str(int(time_so_far // 3600)) + ":" + 
    str(int((time_so_far / 3600 - (time_so_far // 3600)) * 60)) + " minutes.\n")
    file.write("The end day was: " + str(test_month) + "-" + str(test_day) + "-" + str(test_year) + ".\n\n")
    file.write("The neural net models used were :\n")

    nn_tested = False
    for predictor in params["ENSEMBLE"]:
        if "nn" in predictor:
            write_model_params(file, params[predictor], predictor, avg_e[predictor])
        nn_tested = True
    if not nn_tested:
        file.write("NONE\n")

    file.close()

def write_model_params(file, params, predictor, avg_e):
    file.write("Model " + predictor + get_model_name(params))
    file.write("\nParameters: N_steps: " + str(params["N_STEPS"]) + ", Lookup Step:" + str(params["LOOKUP_STEP"]) + ", Test Size: " + str(params["TEST_SIZE"]) + ",\n")
    file.write("Layers: " + layers_string(params["LAYERS"]) + "Test Size: " + str(params["TEST_SIZE"]) + ",\n") 
    file.write("Dropout: " + str(params["DROPOUT"]) + ", Bidirectional: " + str(params["BIDIRECTIONAL"]) + ",\n")
    file.write("Loss: " + params["LOSS"] + ", Optimizer: " + params["OPTIMIZER"] + ", Batch_size: " + str(params["BATCH_SIZE"]) + ",\n")
    file.write("Epochs: " + str(params["EPOCHS"]) + ", Patience: " + str(params["PATIENCE"]) + ", Limit: " + str(params["LIMIT"]) + ".\n")
    file.write("Feature Columns: " + str(params["FEATURE_COLUMNS"]) + "\n")
    file.write("The model used an average of " + str(round(avg_e, 2)) + " epochs.\n\n")
    

def print_backtest_results(params, total_days, avg_p, avg_d, avg_e, year, month, day, time_so_far, current_money, hold_money):
    print("\nTesting finished for ensemble: " + str(params["ENSEMBLE"]))
    print("Using " + str(total_days) + " days, predictions were off by " + avg_p + " percent")
    print("and it predicted the correct direction " + avg_d + " percent of the time ")
    overall_epochs = []
    for predictor in params["ENSEMBLE"]:
        if "nn" in predictor:
            overall_epochs.append(avg_e[predictor])
    all_epochs = mean(overall_epochs)
    print(f"The models (if any) used {all_epochs}.")

    if current_money != None:
        print("If it was trading for real it would have made " + str(current_money) + " as compared to " + str(hold_money) + " if you held it.")
    print("Testing all of the days took " + str(round(time_so_far / 3600, 2)) + " hours or " + str(int(time_so_far // 3600)) + ":" + 
    str(int((time_so_far / 3600 - (time_so_far // 3600)) * 60)) + " minutes.")
    print("The end day was: " + str(month) + "-" + str(day) + "-" + str(year) + "\n")

    nn_tested = False
    for predictor in params["ENSEMBLE"]:
        if "nn" in predictor:
            print_model_params(params[predictor], predictor, avg_e[predictor])
        nn_tested = True
    if not nn_tested:
        print("NONE\n")

def print_model_params(params, predictor, avg_e):
    print("Model " + predictor + get_model_name(params))
    print("Parameters: N_steps: " + str(params["N_STEPS"]) + ", Lookup Step:" + str(params["LOOKUP_STEP"]) + ", Test Size: " + str(params["TEST_SIZE"]) + ",")
    print("Layers: " + layers_string(params["LAYERS"]) + "," + "Test Size: " + str(params["TEST_SIZE"]) + ",")
    print("Dropout: " + str(params["DROPOUT"])  + ", Bidirectional: " + str(params["BIDIRECTIONAL"]) + ",")
    print("Loss: " + params["LOSS"] + ", Optimizer: " +  params["OPTIMIZER"] + ", Batch_size: " + str(params["BATCH_SIZE"]) + ",")
    print("Epochs: " + str(params["EPOCHS"]) + ", Patience: " + str(params["PATIENCE"]) + ", Limit: " + str(params["LIMIT"]) + ".")
    print("Feature Columns: " + str(params["FEATURE_COLUMNS"]))
    print("The model used an average of " + str(avg_e) + " epochs.\n")

def comparator_results_excel(df, run_days, directory, stock):
    lin_avg_p, lin_avg_d, lin_current_money = lin_reg_comparator(df, 14, run_days)
    MA_avg_p, MA_avg_d, MA_current_money = MA_comparator(df, 7, run_days)
    sc_p, sc_d, sc_current_money = smooth_c_comparator(df, 7, 3, run_days)

    directory_string = f"{directory}/{stock}-comparison.txt"
    if not os.path.isfile(directory_string):
        f = open(directory_string, "a")
    else:
        return
    
    f.write("Linear percent away was " + str(lin_avg_p) + " with " + str(lin_avg_d) + " percent prediction making " + str(lin_current_money) + " dollars.\n")
    f.write("Moving average percent away was " + str(MA_avg_p) + " with " + str(MA_avg_d) + " percent prediction making " + str(MA_current_money) + " dollars.\n")
    f.write("sc " + str(sc_p) + " with " + str(sc_d) + " percent prediction making " + str(sc_current_money) + " dollars.")
    f.close()

def read_saved_contents(file_path, return_dict):
    f = open(file_path, "r")

    file_contents = {}
    for line in f:
        parts = line.strip().split("|")
        file_contents[parts[0]] = parts[1]
    f.close()

    for key in file_contents:
        if type(return_dict[key]) == type("str"):
            return_dict[key] = file_contents[key]
        elif type(return_dict[key]) == type(0):
            return_dict[key] = int(file_contents[key])
        elif type(return_dict[key]) == type(0.0):
            return_dict[key] = float(file_contents[key])
        elif type(return_dict[key]) == type([]):
            return_dict[key] = ast.literal_eval(file_contents[key])
        elif type(return_dict[key]) == type({}):
            return_dict[key] = ast.literal_eval(file_contents[key])
        else:
            print("Unexpected type found in this file")
    
    return return_dict

def save_to_dictionary(file_path, dictionary):
    f = open(file_path, "w")

    for key in dictionary:
        f.write(str(key) + "|" + str(dictionary[key]) + "\n")

    f.close()

def plot_graph(y_real, y_pred, symbol, back_test_days, test_var):
    real_y_values = y_real[-back_test_days:]
    predicted_y_values = y_pred[-back_test_days:-1]
    
    plot_dir = directory_dict["graph"] + "/" + get_current_date_string() + "/" + symbol
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    plot_name = plot_dir + "-" + get_time_string() + ".png"
    plt.plot(real_y_values, c="b")
    plt.plot(predicted_y_values, c="r")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title(symbol)
    plt.legend(["Actual Price", "Predicted Price"])
    plt.savefig(plot_name)
    plt.close()

def graph_epochs_relationship(progress, test_name):
        percent_away_list = progress["percent_away_list"]
        correct_direction_list = progress["correct_direction_list"]
        epochs_list = progress["epochs_list"]

        plot_name = directory_dict["graph"] + "/" + test_name + "-dir.png"
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')

        hist, xedges, yedges = np.histogram2d(epochs_list, correct_direction_list, bins=[8, 3], range=[[0, 2000], [0, 1]])

        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dz = hist.ravel()
        c = ["red", "yellow", "green"] * 8
    
        ax.bar3d(xpos, ypos, zpos, 100, .1, dz, color=c, zsort='average')
        ax.set_xlabel("EPOCHS")
        ax.set_ylabel("CORRECT DIR")
        ax.set_zlabel("TIMES IN BUCKET")
        ax.view_init(-2.4, 135)
        
        plt.savefig(plot_name)
        plt.close()

        plot_name = directory_dict["graph"] + "/" + test_name + "-away.png"
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')

        hist, xedges, yedges = np.histogram2d(epochs_list, percent_away_list, bins=[8, 5], range=[[0, 2000], [0, 10]])

        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dz = hist.ravel()
    
        ax.bar3d(xpos, ypos, zpos, 100, 1, dz, zsort='average')
        ax.set_xlabel("EPOCHS")
        ax.set_ylabel("% AWAY")
        ax.set_zlabel("TIMES IN BUCKET")
        ax.view_init(20, 135)
        
        plt.savefig(plot_name)
        plt.close()

