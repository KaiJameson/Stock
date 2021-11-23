import sys
from config.environ import directory_dict
from functions.time_functs import get_past_date_string, increment_calendar, make_Timestamp, read_date_string, get_current_datetime
from functions.trade_functs import get_api
from functions.time_functs import make_Timestamp
from functions.functions import check_directories, get_correct_direction
import datetime


# Option 3: read in multiple tuning files or backtest files
# 
#
#
#

check_directories()

def get_user_input():
    api = get_api()
    if len(sys.argv) > 1:
        if sys.argv[1] == "trade_perform":
            if len(sys.argv) > 2:
                date = read_date_string(sys.argv[2])
                print(f"\n~~~Running trade_perform with date: {date}~~~")
                make_trade_perform_sheet(date, api)
            else:
                print("trade_perform requires another argument for the date.")
                print("Please try again")
                sys.exit(-1)
        elif sys.argv[1] == "PL":
            if len(sys.argv) > 2:
                date = read_date_string(sys.argv[2])
                print(f"\n~~~Running PL with date: {date}~~~")
                make_PL_sheet(date, api)
            else:
                print("PL requires another argument for the date.")
                print("Please try again")
                sys.exit(-1)
        elif sys.argv[1] == "tuning":
            if len(sys.argv) > 2:
                # TODO read given model_name
                make_tuning_sheet()
            else:
                print("Tuning requires another argument for the model name.")
                print("Please try again")
                sys.exit(-1)
        else:
            print("You must give this program one of the three following options.")
            print("\"trade_perform date\" to create the trade performance excel sheet,")
            print("\"PL date\" to create the profit less excel sheet,")
            print("\"tuning model_name\" to create the tuning excel sheet,")
            print("Please try again")
            sys.exit(-1)

    else:
        print("You gotta give some arguments buddy, pick one of the following:")
        print("\"trade_perform date\" to create the trade performance excel sheet,")
        print("\"PL date\" to create the profit less excel sheet,")
        print("\"tuning model_name\" to create the tuning excel sheet,")
        print("Please try again")
        sys.exit(-1)



def make_trade_perform_sheet(date, api):
    if date == get_current_datetime():
        print("You can't run this file with the current date due to the closes not existing yet.")
        print("A file with 2021-11-03 for example is predicting for 2021-11-04, meaning you can't")
        print("compare the runtime/predicted with the actual.")
        print("Please try again")
        sys.exit(-1)
    file_name_ending = f"/{get_past_date_string(date)}.txt"

    trade_perform_file = open(directory_dict["trade_perform"] + file_name_ending, "a")

    runtime_predict_file = open(directory_dict["runtime_predict"] + file_name_ending, "r")
    symbols = runtime_predict_file.readline()
    runtime = runtime_predict_file.readline()
    predict = runtime_predict_file.readline()
    trade_text = symbols + runtime + predict + "\n"
    runtime_predict_file.close()

    symbols = symbols.split(":")
    symbols.pop()
    for i in range(len(symbols)):
        symbols[i] = symbols[i].strip(":\t\n")

    runtime = runtime.split("\t")
    runtime.pop()
    for i in range(len(runtime)):
        runtime[i] = float(runtime[i])

    predict = predict.split("\t")
    predict.pop()
    for i in range(len(predict)):
        predict[i] = float(predict[i])

    date = increment_calendar(date, api, "NA")
    end_date = make_Timestamp(date + datetime.timedelta(1))
    actual_prices = []
    for symbol in symbols:
        barset = api.get_barset(symbol, "day", limit=1, until=end_date)
        current_price = 0
        for symbol, bars in barset.items():
            for bar in bars:
                current_price = bar.c
        actual_prices.append(current_price)

    for i in range(len(runtime)):
        trade_text += str(round(actual_prices[i], 2)) + "\t"
    trade_text += "\n"
    for i in range(len(runtime)):
        trade_text += str(round((abs(actual_prices[i] - predict[i]) / actual_prices[i]) * 100, 2)) + "\t"   
    trade_text += "\n"

    for i in range(len(runtime)):
        trade_text += str(get_correct_direction(predict[i], runtime[i], actual_prices[i])) + "\t"

    trade_perform_file.write(trade_text)
    trade_perform_file.close()
    print("~~~Task Complete~~~")


def make_PL_sheet(date, api):
    if date == get_current_datetime():
        print("You can't run this file with the current date due to the closes not existing yet.")
        print("A file with 2021-11-03 for example is predicting for 2021-11-04, meaning you can't")
        print("compare the runtime/predicted with the actual.")
        print("Please try again")
        sys.exit(-1)
    file_name_ending = f"/{get_past_date_string(date)}.txt"

    pl_file = open(directory_dict["PL"] + file_name_ending, "a")

    runtime_predict_file = open(directory_dict["runtime_predict"] + file_name_ending, "r")
    symbols = runtime_predict_file.readline()
    runtime = runtime_predict_file.readline()
    predict = runtime_predict_file.readline()
    pl_text = symbols
    runtime_predict_file.close()

    runtime_price_file = open(directory_dict["runtime_price"] + file_name_ending, "r")
    runtime = runtime_price_file.readline()
    pl_text += runtime + "\n" + predict + "\n"
    runtime_price_file.close()

    symbols = symbols.split(":")
    symbols.pop()
    for i in range(len(symbols)):
        symbols[i] = symbols[i].strip(":\t\n")

    runtime = runtime.split("\t")
    runtime.pop()
    for i in range(len(runtime)):
        runtime[i] = float(runtime[i])

    predict = predict.split("\t")
    predict.pop()
    for i in range(len(predict)):
        predict[i] = float(predict[i])

    date = increment_calendar(date, api, "NA")
    end_date = make_Timestamp(date + datetime.timedelta(1))
    actual_prices = []
    for symbol in symbols:
        barset = api.get_barset(symbol, "day", limit=1, until=end_date)
        current_price = 0
        for symbol, bars in barset.items():
            for bar in bars:
                current_price = bar.c
        actual_prices.append(current_price)

    for i in range(len(runtime)):
        pl_text += str(round(actual_prices[i], 2)) + "\t"
    pl_text += "\n"

    for i in range(len(runtime)):
        if predict[i] > runtime[i]:
            pl_text += str(round(((actual_prices[i] - runtime[i]) / actual_prices[i]) * 100, 2)) + "\t"   
        else:
            pl_text += "\t"
    pl_text += "\n"

    for i in range(len(runtime)):
        pl_text += str(round(((actual_prices[i] - runtime[i]) / actual_prices[i]) * 100, 2)) + "\t"   
    pl_text += "\n"

    pl_file.write(pl_text)
    pl_file.close()
    print("~~~Task Complete~~~")

def make_tuning_sheet():
    pass

get_user_input()

