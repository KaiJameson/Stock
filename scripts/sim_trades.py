from config.silen_ten import silence_tensorflow
silence_tensorflow()
from config.environ import test_money
from config.symbols import tune_sym_dict
from functions.trade_functs import get_api
from functions.time_functs import get_calendar, get_actual_price
from functions.error_functs import error_handler, keyboard_interrupt
from functions.tuner_functs import increment_and_predict, get_user_input
from functions.data_load_functs import get_proper_df
from functions.time_functs import get_past_datetime
from paca_model import configure_gpu
import time

def simulate_trades(tune_year, tune_month, tune_day, tune_days, params):
    time_s = time.perfcounter()

    api = get_api()
    configure_gpu()
        
    tune_symbols, params = get_user_input(tune_sym_dict, params)

    days_done = 0

    master_df_dict = {}
    portfolio = {
        "cash": test_money,
        "owned_stocks": [],
    }

    try:
        for symbol in tune_symbols:
            master_df_dict[symbol] = get_proper_df(symbol, params["LIMIT"], "V2")

        # tmp_cal = get_calendar(get_past_datetime(tune_year, tune_month, tune_day), api, "AAPL")
        # starting_day_price = get_actual_price((get_past_datetime(tune_year, tune_month, tune_day) 
        #     - datetime.timedelta(1)), master_df_dict[symbol], tmp_cal)

        current_date = get_past_datetime(tune_year, tune_month, tune_day)
        calendar = get_calendar(current_date, api, "AAPL")

        while days_done <= tune_days:
            pred_curr_list = {}
            for symbol in tune_symbols:
                predicted_price, current_price, epochs_run, current_date, sub_df, data_dict = increment_and_predict(symbol, 
                            params, current_date, calendar, master_df_dict[symbol])
                actual_price = get_actual_price(current_date, master_df_dict[symbol], calendar)
                pred_curr_list[symbol] = {"predicted": predicted_price, "current": current_price}
                p_diff = round((abs(actual_price - predicted_price) / actual_price) * 100, 2)

            # do trading shit
                
    
    except KeyboardInterrupt:
        keyboard_interrupt()
    except Exception:
        error_handler(symbol, Exception)