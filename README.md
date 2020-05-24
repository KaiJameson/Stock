#Various information about our program will be listed here

###Setup
* To start you will need to visit [Alpaca](http://alpaca.markets/) and make an account
* After having your account set up you will find a key id and a secret key on the [overview page](https://app.alpaca.markets/brokerage/dashboard/overview)
* Inside of the scripts directory make an api_key.py file and include the following lines inside of it
 * real_api_key_id = '_your key id_'
 * real_api_secret_key = '_your secret key_'

###Types of orders
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
  
###Running Files
* To run scripts you must first make sure to run them from inside of the scripts directory
* In order to run specific stocks and generate their reports, edit symbols=[] inside of decision_tree.py to include the stocks you want to generate reports for and run python decision_tree.py
* In order to tune a stock, edit ticker='' inside of exhaustive_tuning.py and run python exhaustive_tuning.py
