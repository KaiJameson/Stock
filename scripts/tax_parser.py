from api_key import real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key
from functions import check_directories
from environment import tax_directory
import alpaca_trade_api as tradeapi
import pandas as pd
import datetime 
import sys

check_directories()

api = tradeapi.REST(real_api_key_id, real_api_secret_key, base_url="https://api.alpaca.markets")
acts = api.get_activities()


print(acts)
print("Dates printed as M/D/YYYY")


for activity in acts:
    if (activity.activity_type == "FILL"):
      dt = activity.transaction_time.to_pydatetime()
      if (dt.day < 16):
        print(activity.side,activity.qty,"shares of",activity.symbol,"at",activity.price,"on",str(dt.month)+"/"+str(dt.day)+"/"+str(dt.year))

tax_file = sys.argv[1];
print("getting tax info for " + tax_file)

f = open(tax_directory + "/" + tax_file, "r")

dataframe = pd.read_csv(f)

print(dataframe)
dataframe["NetAmount"] = dataframe["NetAmount"].str.replace("$", "").astype(float)
dataframe["Cost"] = dataframe["Cost"].str.replace("$", "").astype(float)
dataframe["Proceed"] = dataframe["Proceed"].str.replace("$", "").astype(float)
print(dataframe["NetAmount"].sum())
print(dataframe["Cost"].sum())
print(dataframe["Proceed"].sum())

