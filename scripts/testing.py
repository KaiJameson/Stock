from api_key import real_api_key_id, real_api_secret_key
import alpaca_trade_api as tradeapi
import numpy as np
import datetime
import time
import pandas as pd
import pytz
import sys
import platform
import math
from time_functions import get_time_string
#assets = api.list_positions()
#print(assets)
# barset = api.get_barset('WTRH', 'day', limit=1000)
# for symbol, bars in barset.items():
#     print(len(bars))
#print(get_time_string())
# api = tradeapi.REST(real_api_key_id, real_api_secret_key)
# barset = api.get_barset('VUZI', 'day', limit=5)
# print("\nDAY\n")
# for symbol, bars in barset.items():
#     print("symbol:", symbol)
#     print("\n")
#     for bar in bars:
#         print(bar.t)
#         print(bar)
        
# print("\n15 MINUTE\n")
# b2 = api.get_barset('ACB', '15Min', limit=2)
# for symbol, bars in b2.items():
#     for bar in bars:
#         print(bar)
#         print(bar.t)
# print("\n1 MINUTE\n")
# b3 = api.get_barset('ACB', '1Min', limit=1)
# for symbol, bars in b3.items():
#     for bar in bars:
#         print(bar)
#         print(bar.t)    
# ts = pd.Timestamp(1589860800, unit='s')
# print(ts)
# now = datetime.datetime.now()
# hour = now.hour
# print('hour', hour)
# date = now.date()

def more_time_stuff():
    tz = 'US/EASTERN'
    now = time.time()
    td = datetime.timedelta(hours=4)
    n = datetime.datetime.fromtimestamp(now)
    rn = n
    hour = rn.hour
    date = rn.date()
    year = date.year
    month = date.month
    day = date.day
    start = datetime.datetime(year, month, day, 9, 15)
    start = time.mktime(start.timetuple())
    t = time.mktime(n.timetuple())
    start = pd.Timestamp(start, unit='s', tz=tz).isoformat()
    end = pd.Timestamp(t, unit='s', tz=tz).isoformat()
    print('hour', hour)
    # n = pd.Timestamp(now, unit='s')
    print(rn)
    print(start)
    print(end)


def time_stuff():
    data = {}
    api = tradeapi.REST(real_api_key_id, real_api_secret_key)
    symbol = ['ZOM']
    tz = 'US/Eastern'
    now = datetime.datetime.now()
    date = now.date()
    year = date.year
    month = date.month
    day = date.day
    start = datetime.datetime(year, month, day, 9, 15)
    start = time.mktime(start.timetuple())
    t = time.mktime(now.timetuple())
    start = pd.Timestamp(start, unit='s', tz=tz).isoformat()
    end = pd.Timestamp(t, unit='s', tz=tz).isoformat()
    barset = api.get_barset(symbol, '15Min', start=start, end=end)
    for symbol, bars in barset.items():
        open_v = bars[0].o
        close = bars[-1].c
        high = 0
        low = math.inf
        for period in bars:
            if period.l < low:
                low = period.l
            if period.h > high:
                high = period.h
        mid = (high + low) / 2
        mid_values = data['mid']
        mid_values.append(mid)
        data['mid'] = mid_values
        open_values = data['open']
        open_values.append(open_v)
        data['open'] = open_values
        high_values = data['high']
        high_values.append(high)
        data['high'] = high_values
        low_values = data['low']
        low_values.append(low)
        data['low'] = low_values
        close_values = data['close']
        close_values.append(close)
        data['close'] = close


# directory = 'information'
# file_name = directory + '/' + get_time_string() + '.txt'
# print('this is the file name:', file_name)
# f = open(file_name, 'w')
# print('file is opened')
# f.write('i can write anything at all\n')
# f.write('and this is proof it still works\n')
# f.close()
# print('file should be closed')

print('testing try excepts')
try:
    ticker = 'hello'
    f_name = ticker + get_time_string
    f = open(f_name, 'r')
except:
    print(sys.exc_info()[1])



