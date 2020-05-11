from rolling_average import RA
import sys


def get_choice():
    try:
        choice = int(input('Enter 1 to read from csv and 2 to enter stocks yourself\n'))
        if choice < 1 or choice > 2:
            return get_choice()
        return choice
    except:
        return get_choice()


def get_day_count():
    try:
        day_count = int(input('enter how many days back you want to grab info for\n'))
        if day_count < 1:
            print('you need to enter a number > 0 ')
            get_day_count()
        return day_count
    except:
        get_day_count()


def get_window(day_count):
    try:
        window = int(input('enter how many days back you want the window to be\n'))
        if window < 1:
            print('you need to enter a number > 0 ')
            get_window(day_count)
        if window > day_count:
            print('your window needs to be at least', day_count, 'which is your day count')
            get_window(day_count)
        return window
    except:
        get_window(day_count)




symbols = []
choice = get_choice()
if choice == 1:
    f = open('stock_list.txt', 'r')
    for line in f:
        symbols.append(line.strip())
    f.close()
elif choice == 2:
    done = False
    while not done:
        ticker = input('Enter the name of a stock to look at. leave blank to exit\n')
        done = ticker.strip() == ''
        if not done:
            symbols.append(ticker.strip().upper())
else:
    print('idk what jus happened, exit time')
    sys.exit(-1)
print('here the symbols you requested to look at', symbols)
cash = 1000
print('starting money is', cash)
day_count = get_day_count()
window = get_window(day_count)
buy = []
sell = []
nothing = []
for symbol in symbols:
    print('for the stock of ticker: ' + symbol)
    try:
        trader = RA(cash, symbol, days=day_count, window=window)
        trader.find_bests()
        # plot_name = 'plots/' + symbol + '_open_close.png'
        # trader.save_plot(plot_name)
        decision = trader.what_to_do()
        if decision == 'sell':
            sell.append(symbol)
        elif decision == 'buy':
            buy.append(symbol)
        else:
            nothing.append(symbol)
    except:
        print('There is an error')
    print('\n--------\n')
print('stocks to buy', buy)
print('stocks to sell', sell)
print('don\'t do anything with the other stocks')
