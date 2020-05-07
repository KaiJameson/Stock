from api_key import paper_api_key_id, paper_api_secret_key
import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def exp_mov_avg(train_data, real_data):
    N = len(train_data)

    run_avg_predictions = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1, N):
        running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)

    print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(real_data)), real_data, color='b', label='True')
    plt.plot(range(0, N), run_avg_predictions, color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()



api = tradeapi.REST(paper_api_key_id, paper_api_secret_key)
#symbols = ['AHPI', 'AMD', 'APDN',  'FARM', 'FNMAT', 'PTKFF', 'WTRH']
symbols = ['WMT']
#symbols = ['AAPL', 'AMD', 'TSLA', 'WMT']
barset = api.get_barset(symbols=symbols, timeframe='day`', limit=300)
data = {}
for symbol, bar in barset.items():
    data[symbol] = [(day.c + day.o)/2 for day in bar]

df = pd.DataFrame(data=data)
#print(df)
std_df = pd.DataFrame.std(df)
stdevs = {}
print('----------')
for key in data:
    info = df[key].to_numpy()
    info = info / info[-1]
    stdev = np.std(info)
    stdevs[key] = stdev
for key in stdevs:
    print(key + ' has a stdev of', stdevs[key])
#testing on AHPI
print('about to test on wmt')
aphi_data = data['WMT']
length = int(len(aphi_data) * .95)
train_data = aphi_data[0:length]
exp_mov_avg(train_data, aphi_data)











