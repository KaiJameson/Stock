from config.api_key import *
from config.environ import stocks_traded
from config.symbols import trading_real_money
from functions.io_functs import make_runtime_price
from functions.error_functs import error_handler
import alpaca_trade_api as tradeapi


def get_api():
    if trading_real_money:
        api = tradeapi.REST(real_api_key_id, real_api_secret_key, base_url="https://api.alpaca.markets")
    else:
        api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")

    return api

def getOwnedStocks():
    api = get_api()
    positions = api.list_positions()
    owned = {}
    for position in positions:
        owned[position.symbol] = position.qty
    return owned

def buy_all_at_once(symbols, owned, price_list):
    api = get_api()
    clock = api.get_clock()
    if not clock.is_open:
        print("\nThe market is closed right now, go home. You're drunk.")
        return

    buy_list = []
    sold_list = {}
    for symbol in symbols:
        try:
            barset = api.get_barset(symbol, "day", limit=1)
            current_price = 0
            for symbol, bars in barset.items():
                for bar in bars:
                    current_price = bar.c
            if current_price < price_list[symbol]["predicted"]:
                if symbol not in owned:
                    buy_list.append(symbol)
                
            else:
                if symbol in owned:
                    qty = owned.pop(symbol)

                    sell = api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )

                    print("\n~~~SELLING " + sell.symbol + "~~~")
                    print("Quantity: " + sell.qty)
                    print("Status: " + sell.status)
                    print("Type: " + sell.type)
                    print("Time in force: "  + sell.time_in_force + "\n")
                    sold_list[symbol] = {"qty":sell.qty, "status":sell.status, "type":sell.type, "time_in_force": sell.time_in_force}

            print("The current price for " + symbol + " is: " + str(round(current_price, 2)))
            make_runtime_price(current_price)

        except Exception:
            error_handler(symbol, Exception)

            
    print("The Owned list: " + str(owned))
    print("The buy list: " + str(buy_list))

    account_equity = float(api.get_account().equity)
    buy_power = float(api.get_account().cash)

    value_in_stocks_before = round((1 - (buy_power / account_equity)) * 100, 2)

    print(f"Value in stocks: {str(value_in_stocks_before)}%")
    print("Account equity: " + str(account_equity))

    stock_portion_adjuster = get_stock_portion_adjuster(value_in_stocks_before, buy_list)
        
    print("\nThe value in stocks is " + str(value_in_stocks_before))
    print("The Stock portion adjuster is " + str(stock_portion_adjuster))

    bought_list = {}
    hold_list = []
    for symbol in symbols:
        try:
            if symbol not in owned and symbol not in buy_list:
                print("~~~Not buying " + symbol + "~~~")
                continue

            elif symbol in owned and symbol not in buy_list:
                print("~~~Holding " + symbol + "~~~")
                hold_list.append(symbol)
                continue
            
            else:
                current_price = 0
                barset = api.get_barset(symbol, "day", limit=1)
                for symbol, bars in barset.items():
                    for bar in bars:
                        current_price = bar.c
                buy_qty = (buy_power / stock_portion_adjuster) // current_price

                if buy_qty == 0:
                    print("Not enough money to purchase stock " + symbol + ".")
                    continue

                buy = api.submit_order(
                    symbol=symbol,
                    qty=buy_qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                
                print("\n~~~Buying " + buy.symbol + "~~~")
                print("Quantity: " + buy.qty)
                print("Status: " + buy.status)
                print("Type: " + buy.type)
                print("Time in force: "  + buy.time_in_force + "\n")
                bought_list[symbol] = {"qty":buy.qty, "status":buy.status, "type":buy.type, "time_in_force":buy.time_in_force}
                
        except Exception:
            error_handler(symbol, Exception)

    account_equity = float(api.get_account().equity)
    buy_power = float(api.get_account().cash)

    value_in_stocks_after= round((1 - (buy_power / account_equity)) * 100, 2)

    return sold_list, hold_list, bought_list, account_equity, value_in_stocks_before, value_in_stocks_after

def get_stock_portion_adjuster(value_in_stocks, buy_list):
    stock_portion_adjuster = 0

    if value_in_stocks > .7:
        stock_portion_adjuster = len(buy_list)
    elif value_in_stocks > .3:
        if (len(buy_list) / stocks_traded) > .8:
            stock_portion_adjuster = len(buy_list) # want 100%
        elif (len(buy_list) / stocks_traded) > .6:
            stock_portion_adjuster = len(buy_list)  # want 100%
        elif (len(buy_list) / stocks_traded) > .4:
            stock_portion_adjuster = len(buy_list) / .90 # want 90%
        else:
            stock_portion_adjuster = len(buy_list) / .70 # want 70%
    else:
        if (len(buy_list) / stocks_traded) > .8:
            stock_portion_adjuster = len(buy_list) # want 100%
        elif (len(buy_list) / stocks_traded) > .6:
            stock_portion_adjuster = len(buy_list) / .90 # want 90%
        elif (len(buy_list) / stocks_traded) > .4:
            stock_portion_adjuster = len(buy_list) / .70 # want 70%
        else:
            stock_portion_adjuster = len(buy_list) / .60 # want 60%

    return stock_portion_adjuster
