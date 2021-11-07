from config.silen_ten import silence_tensorflow
silence_tensorflow()
from functions.functions import layer_name_converter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score
from config.api_key import (real_api_key_id, real_api_secret_key, paper_api_key_id, paper_api_secret_key,
intrinio_sandbox_key, intrinio_production_key)
from config.environ import (test_var, back_test_days, to_plot, test_money, stocks_traded, 
using_all_accuracies, directory_dict)
from config.symbols import trading_real_money
from functions.time_functs import get_time_string, get_past_datetime
from functions.io_functs import make_runtime_price, plot_graph, excel_output, write_nn_report
from functions.error_functs import error_handler
from functions.data_load_functs import load_data
from functions.functions import get_model_name
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from intrinio_sdk.rest import ApiException
import alpaca_trade_api as tradeapi
import talib as ta
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import intrinio_sdk as intrinio
import time
import sys


def nn_report(symbol, total_time, model, data, test_acc, valid_acc, train_acc, N_STEPS, classification):
    time_string = get_time_string()
    # predict the future price
    future_price = predict(model, data, N_STEPS)
    
    y_real, y_pred = return_real_predict(model, data["X_test"], data["y_test"], data["column_scaler"][test_var], classification)

    report_dir = directory_dict["reports_dir"] + "/" + symbol + "/" + time_string + ".txt"
    
    if to_plot:
        plot_graph(y_real, y_pred, symbol, back_test_days, time_string)

    total_minutes = total_time / 60

    real_y_values = y_real[-back_test_days:]
    predicted_y_values = y_pred[-back_test_days:]

    curr_price = real_y_values[-1]
    percent = future_price / curr_price

    write_nn_report(symbol, report_dir, total_minutes, real_y_values, predicted_y_values,
        curr_price, future_price, test_acc, valid_acc, train_acc, y_real, y_pred)
    excel_output(symbol, curr_price, future_price)

    return percent

def create_model(params):
    model = Sequential()
    bi_string = "Bidirectional" if params["BIDIRECTIONAL"] else ""
    # print(bi_string)
    for layer in range(len(params["LAYERS"])):
        if layer == 0:
            model_first_layer(model, params["LAYERS"], layer, params["N_STEPS"])
        elif layer == len(params["LAYERS"]) - 1:
            model_last_layer(model, params["LAYERS"], layer)
        else:
            model_hidden_layers(model, params["LAYERS"], layer)
    
        model.add(Dropout(params["DROPOUT"]))
    model.add(Dense(1, activation="linear"))
    if params["LOSS"] == "huber_loss":
        model.compile(loss=params["LOSS"], metrics=["mean_absolute_error"], optimizer=params["OPTIMIZER"])
    else:
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=params["OPTIMIZER"])
    # print(model.summary())
    return model

def model_first_layer(model, layers, ind, n_steps):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if layer_name == "Dense":
        print("You need to have a recurrent layer leading your model")
        print("otherwise everything breaks, limitation of the loading code.")
        print("Sorry buddy")
        sys.exit(-1)

    if (next_layer_name == "LSTM" or next_layer_name == "SRNN" or next_layer_name == "GRU"):
        model.add(layers[ind][1](layers[ind][0], return_sequences=True, input_shape=(None, n_steps)))
    else:
        model.add(layers[ind][1](layers[ind][0], return_sequences=False, input_shape=(None, n_steps)))

    return model

def model_hidden_layers(model, layers, ind):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if (not(layer_name == "LSTM" or layer_name == "SRNN" or layer_name == "GRU")):
        model.add(layers[ind][1](layers[ind][0]))
    else:
        if (next_layer_name == "LSTM" or next_layer_name == "SRNN" or next_layer_name == "GRU"):
            model.add(layers[ind][1](layers[ind][0], return_sequences=True))
        else:
            model.add(layers[ind][1](layers[ind][0], return_sequences=False))

    return model

def model_last_layer(model, layers, ind):
    layer_name = layer_name_converter(layers[ind])

    if (not(layer_name == "LSTM" or layer_name == "SRNN" or layer_name == "GRU")):
        model.add(layers[ind][1](layers[ind][0]))
    else:
        model.add(layers[ind][1](layers[ind][0], return_sequences=False))
    
    return model

def load_model_with_data(symbol, current_date, params, directory, model_name, to_print=False):
    s = time.time()
    data, train, valid, test = load_data(symbol, params, current_date, shuffle=False, to_print=to_print)
    if to_print:
        print("Loading the data took " + str(time.time() - s) + " seconds")    
    s = time.time()
    model = create_model(params)
    model.load_weights(directory + "/" + params["SAVE_FOLDER"] + "/" + model_name + ".h5")
    if to_print:
        print("Loading the model took " + str(time.time() - s) + " seconds")    

    return data, model

def predict(model, data, n_steps, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][:n_steps]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if not classification:
        predicted_val = column_scaler[test_var].inverse_transform(prediction)[0][0]
    else:
        predicted_val = prediction[0][0]
    return predicted_val

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
    for symbol in symbols:
        try:
            barset = api.get_barset(symbol, "day", limit=1)
            current_price = 0
            for symbol, bars in barset.items():
                for bar in bars:
                    current_price = bar.c
            if current_price < price_list[symbol]:
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

            print("The current price for " + symbol + " is: " + str(round(current_price, 2)))
            make_runtime_price(current_price)

        except Exception:
            error_handler(symbol, Exception)

            
    print("The Owned list: " + str(owned))
    print("The buy list: " + str(buy_list))

    account_equity = float(api.get_account().equity)
    buy_power = float(api.get_account().cash)

    value_in_stocks = 1 - (buy_power / account_equity)

    print("Value in stocks: " + str(value_in_stocks))
    print("Account equity: " + str(account_equity))

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
            

    print("\nThe value in stocks is " + str(value_in_stocks))
    print("The Stock portion adjuster is " + str(stock_portion_adjuster))

    for symbol in symbols:
        try:
            if symbol not in owned and symbol not in buy_list:
                print("~~~Not buying " + symbol + "~~~")
                continue

            elif symbol in owned and symbol not in buy_list:
                print("~~~Holding " + symbol + "~~~")
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
                
        except Exception:
            error_handler(symbol, Exception)

def get_api():
    if trading_real_money:
        api = tradeapi.REST(real_api_key_id, real_api_secret_key, base_url="https://api.alpaca.markets")
    else:
        api = tradeapi.REST(paper_api_key_id, paper_api_secret_key, base_url="https://paper-api.alpaca.markets")

    return api

def sentiment_data(df):
    finviz_url = "https://finviz.com/quote.ashx?t="

    # nltk.download('vader_lexicon')

    time_s = time.time()

    news_tables = {}
    tickers = ["AGYS", "BG"]

    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'}) 
        response = urlopen(req)    
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response, features="lxml")
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[ticker] = news_table

    # Read one single day of headlines for 'AMZN' 
    amzn = news_tables['AGYS']
    # Get all the table rows tagged in HTML with <tr> into 'amzn_tr'
    amzn_tr = amzn.findAll('tr')

    # for i, table_row in enumerate(amzn_tr):
    #     # Read the text of the element 'a' into 'link_text'
    #     a_text = table_row.a.text
    #     # Read the text of the element 'td' into 'data_text'
    #     td_text = table_row.td.text
    #     # Print the contents of 'link_text' and 'data_text' 
    #     print(a_text)
    #     print(td_text)
    #     # Exit after printing 4 rows of data
    #     # if i == 3:
    #     #     break


    parsed_news = []

    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text() 
            # splice text in the td tag into a list 
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                the_time = date_scrape[0]
                
            # else load 'date' as the 1st element and 'time' as the second    
            else:
                date = date_scrape[0]
                the_time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'  
            ticker = file_name.split('_')[0]
            
            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, date, the_time, text])
            
    parsed_news

    vader = SentimentIntensityAnalyzer()

    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']

    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

    print(parsed_and_scored_news)

    plt.rcParams['figure.figsize'] = [10, 6]

    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.groupby(['ticker','date']).mean()

    # Unstack the column ticker
    mean_scores = mean_scores.unstack()

    # Get the cross-section of compound in the 'columns' axis
    mean_scores = mean_scores.xs('compound', axis="columns").transpose()

    # Plot a bar chart with pandas
    mean_scores.plot(kind = 'bar')
    plt.grid()


    print("this took " + str(time.time() - time_s))


def intrinio_news():
    intrinio.ApiClient().set_api_key(intrinio_sandbox_key)
    intrinio.ApiClient().allow_retries(True)

    identifier = "AXP"
    page_size = 1250
    next_page = ""

    response = intrinio.CompanyApi().get_company_news(identifier, page_size=page_size, next_page=next_page)
    print(str(response))

def get_feature_importance(df):
    data = df.copy()
    y = data["close"]
    X = data
   
    train_samples = int(X.shape[0] * 0.8)
 
    X_train_FI = X.iloc[:train_samples]
    X_test_FI = X.iloc[train_samples:]

    y_train_FI = y.iloc[:train_samples]
    y_test_FI = y.iloc[train_samples:]

    regressor = xgb.XGBRegressor(gamma=0.0, n_estimators=150, base_score=0.7, colsample_bytree=1, learning_rate=0.05)
    
    xgbModel = regressor.fit(X_train_FI, y_train_FI, eval_set = [(X_train_FI, y_train_FI), 
    (X_test_FI, y_test_FI)], verbose=False)
    
    fig = plt.figure(figsize=(8,8))
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), 
    tick_label=X_test_FI.columns)
    plt.title('Figure 6: Feature importance of the technical indicators.')
    plt.show()

    feature_names = list(X.columns)
    i = 0
    for feature in xgbModel.feature_importances_.tolist():
        print(feature_names[i], end="")
        print(": "+ str(feature))
        i += 1
    
def get_all_accuracies(model, data, lookup_step, classification=False):
    if using_all_accuracies:
        y_train_real, y_train_pred = return_real_predict(model, data["X_train"], data["y_train"], 
        data["column_scaler"][test_var], classification)
        train_acc = get_accuracy(y_train_real, y_train_pred, lookup_step)
        y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], 
        data["column_scaler"][test_var], classification)
        valid_acc = get_accuracy(y_valid_real, y_valid_pred, lookup_step)
        y_test_real, y_test_pred = return_real_predict(model, data["X_test"], data["y_test"],
         data["column_scaler"][test_var], classification)
        test_acc = get_accuracy(y_test_real, y_test_pred, lookup_step)
    else:
        # print("data X_valid" + str(data["X_valid"]))
        y_valid_real, y_valid_pred = return_real_predict(model, data["X_valid"], data["y_valid"], 
        data["column_scaler"][test_var], classification)
        valid_acc = get_accuracy(y_valid_real, y_valid_pred, lookup_step)
        train_acc = test_acc = 0

    return train_acc, valid_acc, test_acc 

def get_accuracy(y_real, y_pred, lookup_step):
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_real[:-lookup_step], y_pred[lookup_step:]))
    y_real = list(map(lambda current, future: int(float(future) > float(current)), y_real[:-lookup_step], y_real[lookup_step:]))

    return accuracy_score(y_real, y_pred)

def get_all_maes(model, test_tensorslice, valid_tensorslice, train_tensorslice, data):
    train_mae = get_mae(model, train_tensorslice, data)
    valid_mae = get_mae(model, valid_tensorslice, data)
    test_mae =  get_mae(model, test_tensorslice, data)

    return test_mae, valid_mae, train_mae

def get_mae(model, tensorslice, data):
    mse, mae = model.evaluate(tensorslice, verbose=0)
    mae = data["column_scaler"][test_var].inverse_transform([[mae]])[0][0]

    return mae

def return_real_predict(model, X_data, y_data, column_scaler, classification=False):
    y_pred = model.predict(X_data)
    y_real = np.squeeze(column_scaler.inverse_transform(np.expand_dims(y_data, axis=0)))
    if not classification:
        y_pred = np.squeeze(column_scaler.inverse_transform(y_pred))

    return y_real, y_pred

def get_current_price(df):
    return df.close[-1]


