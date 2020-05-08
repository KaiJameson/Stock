from api_key import paper_api_key_id, paper_api_secret_key
import alpaca_trade_api as tradeapi
import numpy as np
from matplotlib import pyplot as plt
api = tradeapi.REST(paper_api_key_id, paper_api_secret_key)

class RA:
    def __init__(self, money, stock_ticket, days, avg_window=6, mean_distance=0.03, window=None):
        self.money = money
        self.stock = stock_ticket
        self.prices = []
        self.roll_avg = []
        self.days = days
        self.avg_window = avg_window
        self.mean_distance = mean_distance
        if window is None:
            self.window = days
        else:
            self.window = window
        self.get_prices()
        self.get_roll_avg()

    def get_prices(self):
        barset = api.get_barset(symbols=self.stock, timeframe='day', limit=self.days)
        for symbol, bar in barset.items():
            self.prices = [(day.c + day.o) / 2 for day in bar]
            #print('bar is ', bar[-1])

    def get_roll_avg(self, avg_window=None):
        if not avg_window is None:
            self.avg_window = avg_window
        avgs = []
        for i in range(self.avg_window, len(self.prices)):
            avgs.append(np.mean(self.prices[i - self.avg_window:i + 1]))
        self.roll_avg = avgs

    def trade(self, prints=False):
        money = self.money
        stocks_owned = 0
        above = self.prices[self.avg_window] > self.roll_avg[self.avg_window]
        start = self.prices[self.avg_window]
        end = self.prices[-1]
        up_bound = 1 + self.mean_distance
        down_bound = 1 - self.mean_distance
        if prints:
            print('starting price is ' + str(start))
            print('ending price is ' + str(end))
            print('above is ' + str(above))
        for i in range(self.avg_window, len(self.prices)):
            if i < len(self.prices) - self.window-1:
                above = self.prices[i] > self.roll_avg[i]
                continue
            price = self.prices[i]
            if above == True and price <= (self.roll_avg[i - self.avg_window] * up_bound):
                above = False
                if stocks_owned > 0:
                    if prints:
                        print('selling')
                    money_made = stocks_owned * price
                    money += money_made
                    stocks_owned = 0
            elif above == False and price >= (self.roll_avg[i - self.avg_window] * down_bound):
                above = True
                num_stocks_to_buy = money // price
                if num_stocks_to_buy > 0:
                    if prints:
                        print('buying')
                        print('price',price)
                    money -= num_stocks_to_buy * price
                    stocks_owned += num_stocks_to_buy
        total_money = money + (stocks_owned * end)
        if prints:
            print('started at ' + str(self.money))
            print('i now have ' + str(total_money) + ' dollars effective')
            print('spencer wanted me to have earned ' + str(self.money * (end / start)))
            print('total money is', total_money)
            print('the money you actually have is', money)
            print('you own this many stocks:', stocks_owned)
        self.money=money
        return total_money



    def limited_trading(self):
        #this function is largely work in progress
        #it returns values lower than the regular trade for now
        money = self.money
        stocks_owned = 0
        stocks_owned_last_checkpoint = 0
        money_last_checkpoint = money
        checkpoint_distance = 30
        above = self.prices[self.avg_window] > self.roll_avg[self.avg_window]
        end = self.prices[-1]
        up_bound = 1 + self.mean_distance
        down_bound = 1 - self.mean_distance
        for i in range(self.avg_window, len(self.prices)):
            if i != self.avg_window and i % checkpoint_distance == 0:
                find_values = Part_RA(money_last_checkpoint, self.prices[i-checkpoint_distance:i+1], stocks_owned_last_checkpoint)
                self.avg_window, self.mean_distance = find_values.find_bests()
                up_bound = 1 + self.mean_distance
                down_bound = 1 - self.mean_distance
                self.get_roll_avg()
                #test a new above here and see if they are different, so you can sell or buy now
                #do i need to do the thing above or is that already handled?
                stocks_owned_last_checkpoint = stocks_owned
                money_last_checkpoint = money
            price = self.prices[i]
            if above == True and price <= (self.roll_avg[i - self.avg_window] * up_bound):
                above = False
                if stocks_owned > 0:
                    money_made = stocks_owned * price
                    money += money_made
                    stocks_owned = 0
            elif above == False and price >= (self.roll_avg[i - self.avg_window] * down_bound):
                above = True
                num_stocks_to_buy = money // price
                if num_stocks_to_buy > 0:
                    money -= num_stocks_to_buy * price
                    stocks_owned += num_stocks_to_buy
        total_money = money + (stocks_owned * end)
        self.money = money
        print('total money is ' + str(total_money))

    def plot(self):
        x = [i for i in range(len(self.prices))]
        plt.figure(figsize=(12, 6))
        plt.plot(x, self.prices, color='blue', label='real data')
        plt.plot(x[self.avg_window:len(x)], self.roll_avg, color='red', label='roll_avg')
        plt.legend()
        plt.show()

    def find_best_average(self):
        starting_money = self.money
        money_list = []
        if self.window == self.days:
            average_list = [i for i in range(2, 14)]
        else:
            average_list = [i for i in range(2, self.window)]
        for i in average_list:
            self.avg_window = i
            self.get_roll_avg()
            money = self.trade()
            money_list.append(money)
            self.money = starting_money
        max_index = money_list.index(max(money_list))
        self.avg_window = average_list[max_index]
        self.get_roll_avg()

    def find_best_mean_distance(self):
        starting_money = self.money
        money_list = []
        mean_list = [(i*0.01) for i in range(10)]
        for i in mean_list:
            self.mean_distance = i
            money = self.trade()
            money_list.append(money)
            self.money = starting_money
        max_index = money_list.index(max(money_list))
        self.mean_distance = mean_list[max_index]
        return money_list[max_index]

    def find_bests(self):
        self.find_best_average()
        max_money =self.find_best_mean_distance()
        print('avg window is', self.avg_window)
        print('mean distance is', self.mean_distance)
        print('max money is', max_money)

    def what_to_do(self):
        above = self.prices[-1] > self.roll_avg[-1]
        print('you are currently above on this stock:', str(above))
        #price = float(input('What is the current price: '))
        barset = api.get_barset(symbols=self.stock, timeframe='1Min', limit=1)
        price = 0
        for symbol, bar in barset.items():
            price = bar[0].c
        print('the current price is', price)
        up_bound = 1 + self.mean_distance
        down_bound = 1 - self.mean_distance
        rolling_avg = self.roll_avg
        rolling_avg.append(np.mean(rolling_avg[len(rolling_avg)-self.avg_window+1:len(rolling_avg)] + [price]))
        if above:
            identifier = price <= rolling_avg[-1]*up_bound
            if identifier:
                print('you should sell')
                return 'sell'
            else:
                print('don\'t sell')
                return 'nothing'
        else:
            identifier = price >= rolling_avg[-1] * down_bound
            if identifier:
                print('you should buy')
                return 'buy'
            else:
                print('don\'t buy')
                return 'nothing'





#test = RA(200, 'AMD', days=252)
#print('limited trading')
#test.limited_trading()
#print('\n------------\n')
# stock = 'MRAM'
# test = RA(200, stock, days=252, avg_window=2, mean_distance=0.05, window=10)
#test.trade(prints=True)
# print('working with stock: ' + stock)
# test.find_bests()
# test.trade(prints=True)
# test.do_i_buy()
#test.plot()
#print('the regular trade')
#test.find_bests()
#symbols = ['AHPI', 'AMD', 'APDN', 'WTRH', 'PENN', 'LYFT', 'NET', 'TRGP', 'IO', 'TUSK', 'VIX', 'ACB']
#symbols = ['TUSK', 'PENN', 'NET', 'LYFT', 'AAPL', 'BLCM']
#symbols = ['MRAM', 'WTRH', 'XERS', 'DOOO', 'SERV']
#symbols = ['THC', 'NWS', 'RDFN', 'NBL', 'CARR', 'CC']
symbols = []
done = False
while not done:
    ticker = input('Enter the name of a stock to look at. leave blank to exit\n')
    done = ticker.strip() == ''
    if not done:
        symbols.append(ticker.strip().upper())
print('here the symbols you requested to look at', symbols)
cash = 100
day_count = 252
for symbol in symbols:
    print('for the stock of ticker: ' + symbol)
    try:
        trader = RA(cash, symbol, days=day_count, window=50)
        trader.find_bests()
        trader.what_to_do()
    except:
        print('There is an error')
    print('\n--------\n')


