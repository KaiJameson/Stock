from alpaca_neural_net import make_neural_net
import os


def read_in_stocks():
    curr_stocks_file = 'owned.csv'
    f = open(curr_stocks_file, 'r')
    i = 0
    money = 0
    stocks_owned = []
    for line in f:
        if i==0:
            money = float(line.strip())
            i += 1
        else:
            stocks_owned.append(line.strip().split(','))
    f.close()
    return money, stocks_owned


def find_percents(symbols):
    percents = {}
    for symbol in symbols:
        percents[symbol] = make_neural_net(symbol)
    return percents


def decide_sells(money, stocks_owned, percents):
    for stock in stocks_owned:
        if percents[stock[0]] <= 1:
            #TODO THE SELL THING
            #the sell thing should include removing from stocks owned and adding the money made from the sell
            continue
    return money, stocks_owned



symbols = ['ZOM', 'PENN', 'WTRH']
directory = 'decisions'
file_name = directory + '/' + 'choices.txt'
if not os.path.isdir(directory):
    os.mkdir(directory)
f = open(file_name, 'w')
percents = find_percents(symbols)
for key in percents:
    f.write(str(percents[key]) + ' ' + key + ' now\n')
f.close()
