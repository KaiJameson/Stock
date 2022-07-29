from config.api_key import paper_api_key_id, paper_api_secret_key
from config.environ import time_zone
from functions.error import  net_error_handler, keyboard_interrupt 
import alpaca_trade_api as tradeapi
import pandas as pd
import platform
import time
import datetime
import copy

def get_start_end():
    operating_sys = platform.system()
    on_linux = operating_sys == 'LINUX'
    now = time.time()
    td = datetime.timedelta(hours=4)
    n = datetime.datetime.fromtimestamp(now)
    if on_linux:
        n -= td
    date = n.date()
    year = date.year
    month = date.month
    day = date.day
    if on_linux:
        start = datetime.datetime(year, month, day, 9, 15) + td
    else:
        start = datetime.datetime(year, month, day, 9, 15)
    start = time.mktime(start.timetuple())
    t = time.mktime(n.timetuple())
    start = pd.Timestamp(start, unit='s', tz=time_zone).isoformat()
    end = pd.Timestamp(t, unit='s', tz=time_zone).isoformat()
    return start, end

def get_time_string():
    operating_sys = platform.system()
    on_linux = operating_sys == 'LINUX'
    now = time.time()
    n = datetime.datetime.fromtimestamp(now)
    if on_linux:
        td = datetime.timedelta(hours=4)
        n -= td
    year = n.year
    month = n.month
    day = n.day
    hour = n.hour
    minute = n.minute
    s = f"{year}-{month}-{day}-{hour}-{minute}"
    return s

def get_current_date_string():
    operating_sys = platform.system()
    on_linux = operating_sys == 'LINUX'
    now = time.time()
    n = datetime.datetime.fromtimestamp(now)
    if on_linux:
        td = datetime.timedelta(hours=4)
        n -= td
    year = n.year
    month = n.month
    day = n.day
    s = f"{year}-{month}-{day}"
    return s

def get_past_date_string(datetime):
    year = datetime.year
    month = datetime.month
    day = datetime.day
    
    return f"{year}-{month}-{day}"


def get_current_datetime():
    now = time.time()
    now = datetime.datetime.fromtimestamp(now)
    
    return now.date()

def get_past_datetime(year, month, day):
    end_date = datetime.datetime(year, month, day)
    
    return end_date.date()

def get_full_end_date():
    now = time.time()
    n = datetime.datetime.fromtimestamp(now)
    date = n.date()
    year = date.year
    month = date.month
    day = date.day
    end_date = datetime.datetime(year, month, day)
    end_date = time.mktime(end_date.timetuple())
    end_date = pd.Timestamp(end_date, unit='s', tz=time_zone).isoformat()
    return end_date


def get_trade_day_back(last_day, days_back):
    api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")
    start = modify_timestamp(-(days_back + days_back * .5), last_day)
    calendar = api.get_calendar(start=start, end=last_day)
    
    reverse_calendar = calendar[::-1]
    trade_day = reverse_calendar[days_back]
    time_int = time.mktime(trade_day.date.timetuple())
    trade_date = pd.Timestamp(time_int, unit='s', tz=time_zone).isoformat()
    return trade_date

def modify_timestamp(days_changed, stamp):
    dt = datetime.datetime.fromisoformat(stamp)
    dt += datetime.timedelta(days_changed)

    new_stamp = make_Timestamp(dt)
    return new_stamp

def get_year_month_day(datetiObj):
    return datetiObj.year, datetiObj.month, datetiObj.day
    
def make_Timestamp(old_date):
    year = old_date.year
    month = old_date.month
    day = old_date.day
    new_date = datetime.datetime(year, month, day)
    time_int = time.mktime(new_date.timetuple())
    new_date = pd.Timestamp(time_int, unit='s', tz=time_zone).isoformat()

    return new_date

def read_date_string(date):
    new_date = datetime.datetime.strptime(date, "%Y-%m-%d")

    return new_date.date()

def get_actual_price(current_date, df, cal):
    df_sub = copy.deepcopy(df)
    cd_copy = copy.copy(current_date)
    # print(f"cd copy before {cd_copy}")
    cd_copy = increment_calendar(cd_copy, cal)
    # print(f"cd copy after inc {cd_copy}")
    df_sub.index = pd.to_datetime(df_sub.index, format="%Y-%m-%d")
    if type(df_sub.index[0]) == type(""):
        df_sub.index = pd.to_datetime(df_sub.index, format="%Y-%m-%d")
        actual_price = df_sub.loc[get_past_date_string(cd_copy)]["c"]
    else:
        actual_price = df_sub.loc[get_past_date_string(cd_copy)]["c"][0]

    del df_sub
    return actual_price

    

def get_calendar(current_date, api, symbol):
    got_cal= False
    while not got_cal:    
        try:
            calendar = api.get_calendar(start=current_date, end=get_current_datetime())
            got_cal = True

        except KeyboardInterrupt:
            keyboard_interrupt()
        except Exception:
            net_error_handler(symbol, Exception)
        
    return calendar

def increment_calendar(current_date, calendar):
    for day, ele in enumerate(calendar):
        if calendar[day].date.date() == current_date:
            current_date = calendar[day + 1].date.date()
            return current_date

    current_date = calendar[0].date.date()
    return current_date
