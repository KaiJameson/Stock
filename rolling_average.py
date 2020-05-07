from api_key import paper_api_key_id, paper_api_secret_key
import alpaca_trade_api as tradeapi
import numpy as np
from matplotlib import pyplot as plt
avg_window = 6
# avg_window = int(6 * (4 * 6.5))
# amount = avg_window * 100
#print('we need to request ' + str(amount))
days_ago = 15
starting_money = 200
api = tradeapi.REST(paper_api_key_id, paper_api_secret_key)
#symbols = ['AHPI', 'AMD', 'APDN',  'FARM', 'FNMAT', 'WTRH', 'NRZ']
symbols = ['AHPI']
barset = api.get_barset(symbols=symbols, timeframe='day', limit=252)
#call based on 0 - 212 with avg_windows of 4-6
data = {}
symbols = []
for symbol, bar in barset.items():
    print('appending symbol ' + symbol)
    symbols.append(symbol)
    data[symbol] = [(day.c + day.o)/2 for day in bar]
    #data[symbol] = [day.c for day in bar]
roll_avgs = {}
for key in data:
    info = np.array(data[key])
    avgs = []
    for i in range(avg_window, len(info)):
        avgs.append(np.mean(info[i-avg_window:i+1]))
    roll_avgs[key] = avgs

money = starting_money
stocks_owned = 0
real_data = data[symbols[0]]
roll_avg = roll_avgs[symbols[0]]
above = real_data[avg_window] > roll_avg[avg_window]
start = real_data[avg_window]
end = real_data[-1]
print('starting price is ' + str(start))
print('ending price is ' + str(end))
print('above is ' + str(above))
for i in range(avg_window, len(real_data)):
    if i < len(real_data) - days_ago:
        above = real_data[avg_window] > roll_avg[avg_window]
        pass
    price = real_data[i]
    if above == True and price <= (roll_avg[i-avg_window] * 1.03):
        above = False
        if stocks_owned > 0:
            print('selling on day ' + str(i))
            money_made = stocks_owned * price
            #print('money made today from selling ' + str(stocks_owned) + ' for ' + str(price))
            money += money_made
            stocks_owned = 0
        # else:
            #print('i wanted to sell stocks for ' + str(price) + ' but i don\'t own any stocks')
    elif above == False and price >= (roll_avg[i-avg_window] * .97):
        above = True
        num_stocks_to_buy = money // price
        if num_stocks_to_buy > 0:
            print('buying on day ' + str(i))
            #print('before buying i have ' + str(money) + ' dollars')
            #print('buying ' + str(num_stocks_to_buy) + ' stocks at ' + str(price) + ' per stock')
            money -= num_stocks_to_buy * price
            stocks_owned += num_stocks_to_buy
            #print('i have ' + str(money) + ' dollars')
        # else:
            #print('i wanted to buy ' + str(num_stocks_to_buy) + ' stocks but didnt have the money')
            #print('i have ' + str(money) + ' dollars')
    # else:
        #print('i held the stocks today')
total_money = money + (stocks_owned * end)
print('started at ' + str(starting_money))
print('i now have ' + str(total_money) + ' dollars effective')
print('spencer wanted me to have earned ' + str(starting_money * (end/start)))
# x = [i for i in range(len(real_data))]
# plt.plot(x, real_data, color='blue', label='real data')
# plt.plot(x[avg_window:len(x)], roll_avg, color='red', label='roll_avg')
# plt.legend()
# plt.show()
