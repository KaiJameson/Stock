from api_key import real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key
from alpaca_neural_net import decision_neural_net
from symbols import decision_symbols
import alpaca_trade_api as tradeapi
import os
import pandas as pd
from time_functions import get_time_string
import threading
import logging
import sys
from environment import stock_decisions_directory, error_file, config_directory
from functions import check_directories
import traceback
api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")

def read_in_stocks(file):
    f = open(file, 'r')
    i = 0
    money = 0
    stocks_owned = []
    for line in f:
        if i==0:
            money = float(line.strip())
            i += 1
        else:
            stocks_owned.append(line.strip().strip(',').split(','))
    f.close()
    return money, stocks_owned


def getOwnedStocks():
    positions = api.list_positions()
    owned = {}
    for position in positions:
        owned[position.symbol] = position.qty
    return owned

def find_percents_and_accs(symbols):
    owned = getOwnedStocks()
    percents = {}
    accuracy = {}
    for symbol in symbols:
        config_name = config_directory + '/' + symbol + '.csv'
        if os.path.isfile(config_name):
            f = open(config_name, 'r')
            values = {}
            for line in f:
                parts = line.strip().split(',')
                values[parts[0]] = parts[1]
            try:
                percents[symbol], accuracy[symbol] = decision_neural_net(symbol,
                    UNITS=int(values['UNITS']), DROPOUT=float(values['DROPOUT']), N_STEPS=int(values['N_STEPS']), EPOCHS=int(values['EPOCHS']))
            #     if accuracy[symbol] >= .7:
            #         try:
            #             qty = owned[symbol]
            #             if percents[symbol] < 1:
            #                 sell = api.submit_order(
            #                     symbol=symbol,
            #                     qty=qty,
            #                     side='sell',
            #                     type='market',
            #                     time_in_force='day'
            #                 )
            #                 print("\nSELLING:", sell)
            #                 print("\n\n")
            #         except KeyError:
            #             if percents[symbol] > 1:
            #                 barset = api.get_barset(symbol, 'day', limit=1)
            #                 current_price = 0
            #                 for symbol, bars in barset.items():
            #                     for bar in bars:
            #                         current_price = bar.c
            #                 if current_price == 0:
            #                     print('\n\nSOMETHING WENT WRONG AND COULDNT GET CURRENT PRICE\n\n')
            #                 else:
            #                     buy_qty = 200 // current_price
            #                     buy = api.submit_order(
            #                         symbol=symbol,
            #                         qty=buy_qty,
            #                         side='buy',
            #                         type='market',
            #                         time_in_force='day'
            #                     )
            #                     print("\nBUYING:", buy)
            #                     print("\n\n")
            except KeyboardInterrupt:
                print('I acknowledge that you want this to stop')
                print('Thy will be done')
                sys.exit(-1)
            except:
                f = open(error_file, 'a')
                f.write('problem with configged stock: ' + symbol + '\n')
                exit_info = sys.exc_info()
                f.write(str(exit_info[1]) + '\n')
                traceback.print_tb(tb=exit_info[2], file=f)
                f.write('listing the dictionary below\n')
                for key in values:
                    f.write(str(key) + ': ' + str(values[key]) + '\n')
                f.close()
                print('\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n')
        else:
            try:
                percents[symbol], accuracy[symbol] = decision_neural_net(symbol)
            except KeyboardInterrupt:
                print('I acknowledge that you want this to stop')
                print('Thy will be done')
                sys.exit(-1)
            except:
                f = open(error_file, 'a')
                f.write('problem with a non configged stock of ticker: ' + symbol + '\n')
                exit_info = sys.exc_info()
                f.write(str(exit_info[1]) + '\n')
                traceback.print_tb(tb=exit_info[2], file=f)
                f.close()
                print('\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n')
    return percents, accuracy


def decide_sells(money, stocks_owned, percents):
    for stock in stocks_owned:
        if percents[stock[0]] <= 1:
            #TODO THE SELL THING
            #the sell thing should include removing from stocks owned and adding the money made from the sell
            continue
    return money, stocks_owned


def read_attributes(file):
    f = open(file, 'r')
    stocks = []
    for line in f:
        stocks.append(line.strip().strip(',').split(','))
    return stocks


check_directories()

symbols = decision_symbols

file_name = stock_decisions_directory + '/' + get_time_string() + '.txt'
if not os.path.isdir(stock_decisions_directory):
    os.mkdir(stock_decisions_directory)
percents, accuracy = find_percents_and_accs(symbols)
f = open(file_name, 'w')
for key in percents:
    f.write(str(percents[key]) + ' ' + key + ' now\n')
    f.write('has an accuracy of ' + str(accuracy[key]) + '\n')
f.close()
