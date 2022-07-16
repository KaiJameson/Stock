from config.silen_ten import silence_tensorflow
silence_tensorflow()
from config.symbols import load_save_symbols, do_the_trades
from config.environ import directory_dict, defaults
from functions.functions import check_directories, r2
from functions.trade import getOwnedStocks, buy_all_at_once
from functions.data_load import get_proper_df, load_all_data, modify_dataframe
from functions.error import error_handler, keyboard_interrupt
from functions.io import  make_load_run_excel, runtime_predict_excel
from functions.time import get_current_date_string
from functions.trade import get_toggleable_api
from functions.prcs_con import pause_running_training, resume_running_training
from paca_model import configure_gpu, ensemble_predictor
import time
import sys



def load_trade(symbols, params, real_mon):
    configure_gpu()

    owned = getOwnedStocks(real_mon)
    print(owned)

    pred_curr_list = {}

    for symbol in symbols:
        try:
            print("\n~~~Now Starting " + symbol + "~~~")
            
            ss = time.perf_counter()
            s = time.perf_counter()

            df = get_proper_df(symbol, params["LIMIT"], "V2")
            
            data_dict = load_all_data(defaults, df)
            print(f"Data processing took {r2(time.perf_counter() - s)} seconds")
            s = time.perf_counter()
            predicted_price, current_price, epochs_dict = ensemble_predictor(symbol, params, None, 
                data_dict, df)    
            print(f"Ensemble_predictor took {r2(time.perf_counter() - s)} seconds")
            
            pred_curr_list[symbol] = {"predicted": predicted_price, "current": current_price}
            print(f"Running everything for {symbol} took {r2(time.perf_counter() - ss)} seconds")
            print("Finished running: " + symbol)

            sys.stdout.flush()

        except KeyboardInterrupt:
            keyboard_interrupt()
        except Exception:
            error_handler(symbol, Exception)

    if do_the_trades:
        time_s = time.perf_counter()
        results = buy_all_at_once(symbols, owned, pred_curr_list, real_mon)
        print("Performing all the trades took " + str(time.perf_counter() - time_s) + " seconds")
    else:
        print("Why are you running this if you don't want to do the trades?")

    runtime_predict_excel(symbols, pred_curr_list)
    create_day_summary(symbols, params, pred_curr_list, results[0], results[1], results[2], results[3], results[4], results[5], real_mon)

def create_day_summary(symbols, params, pred_curr_list, sold_list, hold_list, bought_list, account_equity, value_in_stocks_before, value_in_stocks_after, real_mon):
    api = get_toggleable_api(real_mon)
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



if __name__ == "__main__":
    check_directories()
    s = time.perf_counter()
    if len(sys.argv) > 1:
        if sys.argv[1] == "paper":
            print(f"~~~ Running load_run in paper testing mode ~~~")
            load_trade(load_save_symbols, defaults, False)
    else:
        print(f"~~~ Running load_run in real money mode ~~~")
        pause_list = pause_running_training()
        print(pause_list)
        try:
            load_trade(load_save_symbols, defaults, True)
        except:
            resume_running_training(pause_list)
        resume_running_training(pause_list)
    
    tt = (time.perf_counter() - s) / 60
    print("In total it took " + str(round(tt, 2)) + " minutes to run all the files.")

