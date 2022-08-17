from functions.functions import ra1002, sr2, sr1002, get_correct_direction
from scipy.signal import savgol_filter
from statistics import mean
import talib as ta

def lin_reg_comparator(df, timeperiod, run_days):
    # df = df["df"]
    df["lin_reg"] = ta.LINEARREG(df.c, timeperiod=timeperiod)
    avg_p, avg_d, current_money = simple_one_day_predicting_comparator_guts(df, "lin_reg", run_days)

    return avg_p, avg_d, current_money

def MA_comparator(df, timeperiod, run_days):
    # df = df["df"]
    df["7MA"] = df.c.rolling(window=timeperiod).mean()
    avg_p, avg_d, current_money = simple_one_day_predicting_comparator_guts(df, "7MA", run_days)

    return avg_p, avg_d, current_money

def EMA_comparator(df, timeperiod, run_days):
    # df = df["df"]
    df["EMA"] = ta.EMA(df.c, timeperiod=timeperiod)
    avg_p, avg_d, current_money = simple_one_day_predicting_comparator_guts(df, "EMA", run_days)

    return avg_p, avg_d, current_money

def TSF_comparator(df, timeperiod, run_days):
    # df = df["df"]
    df["TSF"] = ta.TSF(df.c, timeperiod=timeperiod)
    avg_p, avg_d, current_money = simple_one_day_predicting_comparator_guts(df, "TSF", run_days)

    return avg_p, avg_d, current_money

def pre_c_comparator(df, run_days):
    # df = df["df"]
    avg_p, avg_d, current_money = simple_one_day_predicting_comparator_guts(df, "c", run_days)

    return avg_p, avg_d, current_money

def sav_gol_comparator(df, time_period, poly_order, run_days):
    # df = df["df"]
    current_money = 10000
    percent_away_list = []
    correct_direction_list = []

    for i in range(len(df) - 1, len(df) - run_days - 1, -1):
        actual_price = df.c[i]
        df.drop(labels=df.iloc[i].name, inplace=True)
        df["s.c"] = savgol_filter(df.c[:i], time_period, poly_order)
        current_price = df.c[i - 1]
        predicted_price = df.sc[i - 1]

        p_diff = ra1002((actual_price - predicted_price) / actual_price)
        correct_dir = get_correct_direction(predicted_price, current_price, actual_price, "c")
        
        percent_away_list.append(p_diff)
        correct_direction_list.append(correct_dir)
        current_money = update_money(current_money, predicted_price, current_price, actual_price, "c")

    avg_p = sr2(mean(percent_away_list))
    avg_d = sr1002(mean(correct_direction_list))

    return avg_p, avg_d, current_money

def RSI_comparator(df, run_days):
    df = df["df"]
    df["RSI"] = ta.RSI(df.c)

    current_money = 10000
    percent_away_list = []
    correct_direction_list = []

    for i in range(len(df) - 1, len(df) - run_days - 1, -1):
        actual_price = df.c[i]
        current_price = df.c[i - 1]
        predicted_price = df.c[i - 1] * (1/150000 * (-df.RSI[i - 1] + 50)**3 + 1)
            
        p_diff = ra1002((actual_price - predicted_price) / actual_price)
        correct_dir = get_correct_direction(predicted_price, current_price, actual_price, "c")
        
        percent_away_list.append(p_diff)
        correct_direction_list.append(correct_dir)
        current_money = update_money(current_money, predicted_price, current_price, actual_price, "c")
    
    avg_p = sr2(mean(percent_away_list))
    avg_d = sr1002(mean(correct_direction_list))

    return avg_p, avg_d, current_money

def simple_one_day_predicting_comparator_guts(df, comp, run_days):
    current_money = 10000
    percent_away_list = []
    correct_direction_list = []

    for i in range(len(df) - 1, len(df) - run_days - 1, -1):
        actual_price = df.c[i]
        current_price = df.c[i - 1]
        predicted_price = df[comp][i - 1]

        p_diff = ra1002((actual_price - predicted_price) / actual_price)
        correct_dir = get_correct_direction(predicted_price, current_price, actual_price, "c")
        
        percent_away_list.append(p_diff)
        correct_direction_list.append(correct_dir)
        current_money = update_money(current_money, predicted_price, current_price, actual_price, "c")

    avg_p = sr2(mean(percent_away_list))
    avg_d = sr1002(mean(correct_direction_list))

    return avg_p, avg_d, current_money

def update_money(current_money, predicted_value, current_price, actual_price, test_var):
    p_change = 1 + ((actual_price - current_price) / actual_price)
    stocks = 0
    if test_var == "acc":
        print("getting there")
        print(f"predicted value {predicted_value}")
        if predicted_value == 1:
            stocks = current_money // current_price
            if stocks > 0:
                current_money -= stocks * current_price
                current_money += stocks * current_price * p_change
    else:
        if predicted_value > current_price:
            stocks = current_money // current_price
            if stocks > 0:
                current_money -= stocks * current_price
                current_money += stocks * current_price * p_change

    return round(current_money, 2)
