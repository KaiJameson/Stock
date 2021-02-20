import datetime 

class BuyOrder:
  def __init__(self, ticker, qty, price, date):
    self.ticker = ticker 
    self.qty = qty 
    self.price = price 
    self.date = date 
  
  def dateAcquired(self):
    return str(self.date.month) + "/" + str(self.date.day) + "/" + str(self.date.year)

  def canSellAll(self, qty):
    return self.qty >= qty 
  
  def quantity(self):
    return self.qty 
  
  def sell(self, qty, price, date):
    sellDate = self.dateSold(date)
    profit = (qty * price) - qty * (self.price)
    self.qty -= qty 
    return self.dateAcquired() + "," + sellDate + "," + self.ticker + "," + str(qty*self.price) + "," + str(qty*price) + "," + format(profit, ".2f")

  @staticmethod 
  def dateSold(date):
    return str(date.month) + "/" + str(date.day) + "/" + str(date.year)













