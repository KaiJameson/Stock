from api_key import real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key
import alpaca_trade_api as tradeapi
import datetime 
api = tradeapi.REST(real_api_key_id, real_api_secret_key, base_url="https://api.alpaca.markets")
acts = api.get_activities()
print("Dates printed as M/D/YYYY")
for activity in acts:
    if (activity.activity_type == "FILL"):
      dt = activity.transaction_time.to_pydatetime()
      if (dt.year < 2021):
        print(activity.side,activity.qty,"shares of",activity.symbol,"at",activity.price,"on",str(dt.month)+"/"+str(dt.day)+"/"+str(dt.year))

        
