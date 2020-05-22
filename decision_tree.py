from alpaca_neural_net import make_neural_net
import os
import pandas as pd
from time_functions import get_time_string
import threading
import logging





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


def find_percents_and_accs(symbols):
    percents = {}
    accuracy = {}
    for symbol in symbols:
        error_file = 'error_file.txt'
        config_name = 'config/' + symbol + '.csv'
        if os.path.isfile(config_name):
            f = open(config_name, 'r')
            values = {}
            for line in f:
                parts = line.strip().split(',')
                values[parts[0]] = parts[1]
            try:
                percents[symbol], accuracy[symbol] = make_neural_net(symbol,
                    UNITS=int(values['UNITS']), DROPOUT=float(values['DROPOUT']), N_STEPS=int(values['N_STEPS']), EPOCHS=int(values['EPOCHS']))
            except:
                f = open(error_file, 'a')
                f.write('problem with configged stock: ' + symbol + '\n')
                f.write('listing the dictionary below\n')
                for key in values:
                    f.write(str(key) + ': ' + str(values[key]) + '\n')
                f.close()
        else:
            try:
                percents[symbol], accuracy[symbol] = make_neural_net(symbol)
            except:
                f = open(error_file, 'a')
                f.write('problem with a non configged stock of ticker: ' + symbol + '\n')
                f.close()
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


symbols = ['DOOO', 'ACB', 'LIVX']
directory = 'information'
file_name = directory + '/' + get_time_string() + '.txt'
if not os.path.isdir(directory):
    os.mkdir(directory)
percents, accuracy = find_percents_and_accs(symbols)
f = open(file_name, 'w')
for key in percents:
    f.write(str(percents[key]) + ' ' + key + ' now\n')
    f.write('has an accuracy of ' + str(accuracy[key]) + '\n')
f.close()
