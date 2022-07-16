import numpy as np
import math


def garman_klass(name, df, window=30, trading_periods=252):

    log_hl = (df['h'] / df['l']).apply(np.log)
    log_co = (df['c'] / df['o']).apply(np.log)

    rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
    
    def f(v):
        return (trading_periods * v.mean())**0.5
    
    df[name] = rs.rolling(window=window, center=False).apply(func=f)
    

def hodges_tompkins(name, df, window=30, trading_periods=252):
    
    log_return = (df['c'] / df['c'].shift(1)).apply(np.log)

    vol = log_return.rolling(
        window=window,
        center=False
    ).std() * math.sqrt(trading_periods)

    h = window
    n = (log_return.count() - h) + 1

    adj_factor = 1.0 / (1.0 - (h / n) + ((h**2 - 1) / (3 * n**2)))

    df[name] = vol * adj_factor


def get_kurtosis(name, df, window=30):

    log_return = (df['c'] / df['c'].shift(1)).apply(np.log)

    df[name] = log_return.rolling(
        window=window,
        center=False
    ).kurt()


def parkinson(name, df, window=30, trading_periods=252):

    rs = (1.0 / (4.0 * math.log(2.0))) * ((df['h'] / df['l']).apply(np.log))**2.0

    def f(v):
        return (trading_periods * v.mean())**0.5
    
    df[name] = rs.rolling(
        window=window,
        center=False
    ).apply(func=f)
    
def rogers_stachell(name, df, window=30, trading_periods=252):
    
    log_ho = (df['h'] / df['o']).apply(np.log)
    log_lo = (df['l'] / df['o']).apply(np.log)
    log_co = (df['c'] / df['o']).apply(np.log)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return (trading_periods * v.mean())**0.5
    
    df[name] = rs.rolling(
        window=window,
        center=False
    ).apply(func=f)
    

def get_skew(name, df, window=30):

    log_return = (df['c'] / df['c'].shift(1)).apply(np.log)
    
    df[name] = log_return.rolling(
        window=window,
        center=False
    ).skew()


def yang_zhang(name, df, window=30, trading_periods=252):

    log_ho = (df['h'] / df['o']).apply(np.log)
    log_lo = (df['l'] / df['o']).apply(np.log)
    log_co = (df['c'] / df['o']).apply(np.log)
    
    log_oc = (df['o'] / df['c'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (df['c'] / df['c'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    c_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    df[name] = (open_vol + k * c_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

