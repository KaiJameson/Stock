from config.silen_ten import silence_tensorflow
silence_tensorflow()
from config.symbols import load_save_symbols, do_the_trades
from config.environ import directory_dict, defaults
from functions.functions import check_directories, get_model_name, percent_from_real
from functions.trade_functs import getOwnedStocks, buy_all_at_once
from functions.data_load_functs import load_data
from functions.error_functs import error_handler
from functions.io_functs import  make_load_run_excel, runtime_predict_excel
from functions.time_functs import get_current_date_string
from functions.trade_functs import get_api
from paca_model import configure_gpu, ensemble_predictor
import tensorflow as tf
import psutil
import time
import sys

check_directories()

def load_trade(symbols, params):
    configure_gpu()

    owned = getOwnedStocks()

    pred_curr_list = {}

    for symbol in symbols:
        try:
            print("\n~~~Now Starting " + symbol + "~~~")
            
            s = time.perf_counter()
            predicted_price, current_price, epochs_dict = ensemble_predictor(symbol, params, None)    
            print(f"Ensemble_predictor took {round(time.perf_counter() - s, 2)}")
            
            pred_curr_list[symbol] = {"predicted": predicted_price, "current": current_price}

            print("Finished running: " + symbol)

            sys.stdout.flush()

        except KeyboardInterrupt:
            print("I acknowledge that you want this to stop")
            print("Thy will be done")
            sys.exit(-1)
        except Exception:
            error_handler(symbol, Exception)

    if do_the_trades:
        time_s = time.perf_counter()
        results = buy_all_at_once(symbols, owned, pred_curr_list)
        print("Performing all the trades took " + str(time.perf_counter() - time_s) + " seconds")
    else:
        print("Why are you running this if you don't want to do the trades?")

    runtime_predict_excel(symbols, pred_curr_list)
    create_day_summary(symbols, params, pred_curr_list, results[0], results[1], results[2], results[3], results[4], results[5])

def create_day_summary(symbols, params, pred_curr_list, sold_list, hold_list, bought_list, account_equity, value_in_stocks_before, value_in_stocks_after):
    api = get_api()
    account = api.get_account()
    day_text = "~~~ Symbols sold, hold, or bought ~~~\n"
    for symbol in sold_list:
        day_text += ("Sold: " + symbol + str(sold_list[symbol]) + "\n")
    for symbol in hold_list:
        pos = api.get_position(symbol)
        # print(pos)
        day_text += (f"Held:{symbol} qty:{str(pos.qty)} value:{str(pos.market_value)} avg_entry:{str(pos.avg_entry_price)} change_today:{str(round(float(pos.change_today) * 100, 2))} unrealized_PL:{str(pos.unrealized_intraday_pl)}\n")
    for symbol in bought_list:
        day_text += ("Bought: " + symbol + str(bought_list[symbol]) + "\n")
    day_text += f"\nStarted the day with {account.last_equity} and ended it with {account_equity}.\n"
    day_text += f"{value_in_stocks_before}% of the portfolio was in stocks before buying and {value_in_stocks_after}% was in after.\n"
    # for symbol in symbols:
        
    #     if to_plot:
    #         # plot_graph
    #         pass

    #     train_acc, valid_acc, test_acc = get_all_accuracies(model, data, defaults["LOOKUP_STEP"])
    #     print("Getting the accuracies took " + str(time.perf_counter() - time_s) + " seconds")   
    #     y_real, y_pred = return_real_predict(model, data["X_valid"], data["y_valid"], data["column_scaler"]["c"])
    #     make_load_run_excel(symbol, train_acc, valid_acc, test_acc, percent_from_real(y_real, y_pred), abs((percent - 1) * 100))

    sum_f = open(directory_dict["day_summary"] + "/" + get_current_date_string() + ".txt", "a")
    sum_f.write(day_text)
    sum_f.close()


def pause_running_training():
    processes = {p.pid: p.info for p in psutil.process_iter(["name"])}
    
    python_processes_pids = []
    pause_list = []

    for process in processes:
        if processes[process]["name"] == "python.exe":
            python_processes_pids.append(process)

    for pid in python_processes_pids:
        if any("batch" in string for string in psutil.Process(pid).cmdline()):
            pause_list.append(pid)
        elif any("tuning" in string for string in psutil.Process(pid).cmdline()):
            pause_list.append(pid)

    for pid in pause_list:
        psutil.Process(pid).suspend()

    return pause_list

def resume_running_training(pause_list):
    print("hello")
    for pid in pause_list:
        print(pid)
        psutil.Process(pid).resume()

if __name__ == "__main__":
    s = time.perf_counter()
    pause_list = pause_running_training()
    load_trade(load_save_symbols, defaults)
    resume_running_training(pause_list)
    tt = (time.perf_counter() - s) / 60
    print("In total it took " + str(round(tt, 2)) + " minutes to run all the files.")

