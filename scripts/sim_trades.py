from config.silen_ten import silence_tensorflow
silence_tensorflow()
from config.environ import test_money, directory_dict
from config.symbols import sym_dict, sim_trades_dict, tune_year, tune_month, tune_day, tune_days
from config.model_repository import models
from functions.trade import get_api, more_than_X, preport_no_rebal, random_guess, rebal_split, top_X, more_than_X, no_buy_if_less_than_X
from functions.time import get_calendar, increment_calendar, get_actual_price
from functions.error import error_handler, keyboard_interrupt
from functions.tuner import subset_and_predict, get_user_input
from functions.paca_model import get_current_price
from functions.data_load import get_proper_df, df_subset
from functions.time import get_past_datetime
from functions.functions import check_directories, r2, interpret_dict
from paca_model import configure_gpu
from statistics import mean
import datetime
import time
import sys


        

def simulate_trades(tune_year, tune_month, tune_day, tune_days, params):
    time_s = time.perf_counter()

    api = get_api()
    configure_gpu()
        
    tune_symbols, params = get_user_input(sym_dict, params)

    days_done = 1

    master_df_dict = {}
    portfolio = {
        "cash": test_money,
        "equity": test_money,
        "owned": {},
        "buy_prices": {},
    }

    symbol = "SPY"
    try:
        tmp_cal = get_calendar(get_past_datetime(tune_year, tune_month, tune_day), api, symbol)
        spy_df = get_proper_df(symbol, params["LIMIT"], "V2")
        spy_start_price = get_actual_price((get_past_datetime(tune_year, tune_month, tune_day) 
            - datetime.timedelta(1)), spy_df, tmp_cal)

        qqq_df = get_proper_df("QQQ", params["LIMIT"], "V2")
        qqq_start_price = get_actual_price((get_past_datetime(tune_year, tune_month, tune_day) 
            - datetime.timedelta(1)), qqq_df, tmp_cal)

        current_date = get_past_datetime(tune_year, tune_month, tune_day)
        starting_prices = []
        for symbol in tune_symbols:
            master_df_dict[symbol] = get_proper_df(symbol, params["LIMIT"], "V2")
            starting_prices.append(get_actual_price((get_past_datetime(tune_year, tune_month, tune_day) 
            - datetime.timedelta(1)), master_df_dict[symbol], tmp_cal))
            
        print(starting_prices)

        calendar = get_calendar(current_date, api, "AAPL")

        while days_done <= tune_days:
            time_d = time.perf_counter()
            print(f"\nCurrently on day {days_done} of {tune_days} \n")
            print(current_date)

            pred_curr_list = {}
            current_date = increment_calendar(current_date, calendar)
            
            for symbol in tune_symbols:
                predicted_price, current_price, epochs_run, sub_df, data_dict = subset_and_predict(symbol, 
                            params, current_date, master_df_dict[symbol], to_print=False)
                pred_curr_list[symbol] = {"predicted": predicted_price, "current": current_price}


            # recalculate equity
            portfolio["equity"] = 0.0
            for symbol in portfolio["owned"]:
                portfolio["equity"] += portfolio["owned"][symbol]["qty"] * pred_curr_list[symbol]["current"]
            portfolio["equity"] += portfolio["cash"]
            
            if params["TRADE_METHOD"] == "preport_no_rebal":
                portfolio = preport_no_rebal(tune_symbols, pred_curr_list, portfolio)
            elif params["TRADE_METHOD"] == "rebal_split":
                portfolio = rebal_split(tune_symbols, pred_curr_list, portfolio)
            elif params["TRADE_METHOD"] == "top_X":
                portfolio = top_X(tune_symbols, pred_curr_list, portfolio, params["TRADE_PARAMS"])
            elif params["TRADE_METHOD"] == "more_than_X":
                portfolio = more_than_X(tune_symbols, pred_curr_list, portfolio, params["TRADE_PARAMS"])
            elif params['TRADE_METHOD'] == "random_guess":
                portfolio = random_guess(tune_symbols, pred_curr_list, portfolio)
            elif params['TRADE_METHOD'] == "no_buy_if_less_than_X":
                portfolio = no_buy_if_less_than_X(tune_symbols, pred_curr_list, portfolio, params["TRADE_PARAMS"])
            else:
                print(f"Don't have trading strategy {params['TRADE_METHOD']} implemented yet")
                print(f"Sorry bud, try again next time")
                sys.exit(-1)

            print(f"""End of day ... equity:{r2(portfolio["equity"])} cash:{r2(portfolio["cash"])}\n""", flush=True)
            
            print(portfolio["owned"])
            days_done += 1
            print(f"Running this day took {r2((time.perf_counter() - time_d))} seconds")
    
    except KeyboardInterrupt:
        keyboard_interrupt()
    except Exception:
        error_handler(symbol, Exception)
    

    spy_sub_df = df_subset(current_date, spy_df)
    spy_end_price = get_current_price(spy_sub_df)

    qqq_sub_df = df_subset(current_date, qqq_df)
    qqq_end_price = get_current_price(qqq_sub_df)

    current_prices = []
    for symbol in pred_curr_list:
        current_prices.append(pred_curr_list[symbol]["current"])

    
    time_so_far = time.perf_counter() - time_s

    print(f"\nTesting finished for ensemble:{params['ENSEMBLE']} using trade method:{params['TRADE_METHOD']} with params:{interpret_dict(params['TRADE_PARAMS'])}")
    print(f"The total value of the portfolio was {r2(portfolio['equity'])} at the end with {r2(portfolio['cash'])} in cash")
    print(f"Holding this group of stocks would have made {r2(test_money * (mean(current_prices) / mean(starting_prices)))}\n"
        f"Holding the S&P would have made {r2(test_money * (spy_end_price / spy_start_price))}\n"
        f"Holding the NASDAQ would have made {r2(test_money * (qqq_end_price / qqq_start_price))}")
    print(f"It was holding these {portfolio['owned']} stocks at the end")
    print(f"Testing all of the days took {r2(time_so_far / 3600)} hours or {int(time_so_far // 3600)}:"
        f"{int((time_so_far / 3600 - (time_so_far // 3600)) * 60)} minutes.")

    print(interpret_dict(params['TRADE_PARAMS']))
    with open(f"{directory_dict['sim_trades']}/{params['ENSEMBLE']}-{params['TRADE_METHOD']}-{interpret_dict(params['TRADE_PARAMS'])}.txt", "a") as f:
        f.write(f"\nTesting finished for ensemble:{params['ENSEMBLE']} using trade method:{params['TRADE_METHOD']} with params:{interpret_dict(params['TRADE_PARAMS'])}\n")
        f.write(f"The total value of the portfolio was {r2(portfolio['equity'])} at the end with {r2(portfolio['cash'])} in cash\n")
        f.write(f"Holding this group of stocks would have made {r2(test_money * (mean(current_prices) / mean(starting_prices)))}.\n"
            f"Holding the S&P would have made {r2(test_money * (spy_end_price / spy_start_price))}.\n"
            f"Holding the NASDAQ would have made {r2(test_money * (qqq_end_price / qqq_start_price))}.\n")
        f.write(f"It was holding these {portfolio['owned']} stocks at the end\n")
        f.write(f"Testing all of the days took {r2(time_so_far / 3600)} hours or {int(time_so_far // 3600)}:"
            f"{int((time_so_far / 3600 - (time_so_far // 3600)) * 60)} minutes.\n")


if __name__ == "__main__":
    check_directories()
    

    for model in sim_trades_dict["ENSEMBLE"]:
        if model in models:
            sim_trades_dict[model] = models[model]

    simulate_trades(tune_year, tune_month, tune_day, tune_days, sim_trades_dict)
    
