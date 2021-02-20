from api_key import real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key
import alpaca_trade_api as tradeapi
import datetime 
from BuyOrder import BuyOrder
api = tradeapi.REST(real_api_key_id, real_api_secret_key, base_url="https://api.alpaca.markets")
activities = api.get_activities()
stocks = {}
sellList = []
for activity in activities[::-1]:
    if (activity.activity_type == "FILL"):
      if activity.side == "buy":
        order = BuyOrder(activity.symbol, int(activity.qty), float(activity.price), activity.transaction_time.to_pydatetime())
        if not activity.symbol in stocks:
          stocks[activity.symbol] = [order]
        else:
          stocks[activity.symbol].append(order)
      elif activity.side == "sell":
        dt = activity.transaction_time.to_pydatetime()
        buyOrders = stocks[activity.symbol]
        qty = int(activity.qty) 
        while qty > 0:
          for i in range(len(buyOrders)):
            buyQty = buyOrders[i].quantity()
            if buyQty < qty:
              sellList.append(buyOrders[i].sell(buyQty, float(activity.price), dt))
              qty -= buyQty 
            else:
              sellList.append(buyOrders[i].sell(qty, float(activity.price), dt))
              qty = 0
      # if (dt.year < 2021):
        # print(activity.side,activity.qty,"shares of",activity.symbol,"at",activity.price,"on",str(dt.month)+"/"+str(dt.day)+"/"+str(dt.year))
f = open("tax_info.csv", 'w')
f.write("dateAcquired,dateSold,Symbol,cost,proceed,netAmount\n")
for sell in sellList:
  f.write(sell + '\n')
f.close()        


def find_profit():
  f = open("tax_info.csv", 'r')
  f.readline()
  profit = 0
  for line in f:
    columns = line.strip().split(',')
    yearSold = columns[1].split('/')[2]
    if yearSold == '2020':
      profit += float(columns[-1])

  f.close()
  print(profit)

find_profit()


