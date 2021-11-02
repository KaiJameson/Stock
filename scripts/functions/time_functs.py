from config.api_key import paper_api_key_id, paper_api_secret_key
from config.environ import time_zone
from functions.error_functs import  net_error_handler
import alpaca_trade_api as tradeapi
import pandas as pd
import platform
import time
import datetime

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

def get_date_string():
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
    s = f"{year}-{month}-{day}"
    return s

def zero_pad_date_string():
    operating_sys = platform.system()
    on_linux = operating_sys == 'LINUX'
    now = time.time()
    now = datetime.datetime.fromtimestamp(now)
    if on_linux:
        td = datetime.timedelta(hours=4)
        now -= td

    padded = datetime.date(now.year, now.month, now.day) + datetime.timedelta(1)
    return str(padded)

def get_short_end_date(year, month, day):
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
    calendar = api.get_calendar(end=last_day)
    reverse_calendar = calendar[::-1]
    trade_day = reverse_calendar[days_back]
    time_int = time.mktime(trade_day.date.timetuple())
    trade_date = pd.Timestamp(time_int, unit='s', tz=time_zone).isoformat()
    return trade_date

def get_year_month_day(datetiObj):
    return datetiObj.year, datetiObj.month, datetiObj.day

def increment_calendar(current_date, api, symbol):
    date_changed = False
    while not date_changed:    
        try:
            time_s = time.time()
            calendar = api.get_calendar(start=current_date + datetime.timedelta(1), end=current_date + datetime.timedelta(1))[0]
            while calendar.date != current_date + datetime.timedelta(1):
                print("Skipping " + str(current_date + datetime.timedelta(1)) + " because it was not a market day.")
                current_date = current_date + datetime.timedelta(1)

            print("\nMoving forward one day in time: \n")
            
            current_date = current_date + datetime.timedelta(1)
            date_changed = True
                
        except Exception:
            if date_changed:
                current_date = current_date - datetime.timedelta(1)
            net_error_handler(symbol, Exception)

    return current_date
    
def make_Timestamp(old_date):
    year = old_date.year
    month = old_date.month
    day = old_date.day
    new_date = datetime.datetime(year, month, day)
    time_int = time.mktime(new_date.timetuple())
    new_date = pd.Timestamp(time_int, unit='s', tz=time_zone).isoformat()

    return new_date

def get_actual_price(current_date, api, symbol):
    no_price = True
    while no_price:
        try:
            calendar = api.get_calendar(start=current_date + datetime.timedelta(1), end=current_date + datetime.timedelta(1))[0]
            one_day_in_future = make_Timestamp(calendar.date + datetime.timedelta(1))
            barset = api.get_barset(symbols=symbol, timeframe="day", limit=1, until=one_day_in_future)
            for symbol, bars in barset.items():
                for bar in bars:
                    actual_price = bar.c
            no_price = False

        except Exception:
            net_error_handler(symbol, Exception)

    return actual_price


