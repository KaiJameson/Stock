# Various information about our program will be listed here

### Setup
1. To start you will need to visit [Alpaca](http://alpaca.markets/) and make an account
2. After having your account set up you will find a key id and a secret key on the [overview page](https://app.alpaca.markets/brokerage/dashboard/overview)
3. Inside of the scripts directory make an api_key.py file and include the following lines inside of it
 * real_api_key_id = "_your key id_"
 * real_api_secret_key = "_your secret key_"
4. Install ta-lib with a whrl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### List of Dependancies You Need to Install
* tensorflow
* sklearn
* pandas
* nltk
* alpaca_trade_api
* matplotlib
* intrinio_sdk
* numpy
* bs4

### Settings
* Most of the necessary settings are included in the environment.py file, however to get load_run and save_train to run you will
need to make a load_save_symbols list with the stocks that you want to run
 * The same applies for all of the other files like decision tree and real test
* You also need a do_the_trades bool to control whether or not you trade and a trading_real_money bool for whether to use
alpaca paper or real account

### Types of orders
* Market order: fill the order right now at whatever price
* Limit 
 * Buy: wait till the price drops below this point and then START to buy (not guaranteed to get them all)
 * Sell: wait till the price rises above this point and then START to sell
* Stop:
 * Buy: wait till the price goes above this point and then buy it, as a market order (so should fill immediately)
 * Sell: if the price goes below this point, sell all (this is also called a stop loss)
* Stop Limit:
 * Buy: After the price drops below this stop point, submit a limit order at the limit price
 * Sell: After the price goes above this stop point, submit a limit order at the limit price
  
### Running Files
* To run scripts you must first make sure to run them from inside of the scripts directory
* In order to run specific stocks and generate their reports, edit load_trade_symbols=[] inside of 
symbols.py to include the stocks you want. You will then first run the train_save.py to generate and save the models 
and then later you can run that model with load_run.py
* In order to tune a stock, edit ticker="" inside of exhaustive_tuning.py and run python exhaustive_tuning.py


### Owned Stocks
* A history of of your buys/sells can be accessed through alpaca's api
