import platform
import time
import datetime
import pandas as pd


def get_start_end():
    operating_sys = platform.system()
    on_linux = operating_sys == 'LINUX'
    tz = 'US/EASTERN'
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
        start = datetime.datetime(year, month, day, 9, 15)+ td
    else:
        start = datetime.datetime(year, month, day, 9, 15)
    start = time.mktime(start.timetuple())
    t = time.mktime(n.timetuple())
    start = pd.Timestamp(start, unit='s', tz=tz).isoformat()
    end = pd.Timestamp(t, unit='s', tz=tz).isoformat()
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
