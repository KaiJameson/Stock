from paca_model import saveload_neural_net
from symbols import load_save_symbols
from functions import check_directories
from error_functs import error_handler
from environ import defaults
import sys
import time

check_directories()

def save_models(symbols):
    for symbol in symbols:
        try:
            print(f"\n~~~ Now Training {symbol} ~~~")
            epochs = saveload_neural_net(symbol, end_date=None, params=defaults)

        except KeyboardInterrupt:
            print("I acknowledge that you want this to stop.")
            print("Thy will be done.")
            sys.exit(-1)
        except Exception:
            error_handler(symbol, Exception)
        
s = time.time()
save_models(load_save_symbols)
end = (time.time() - s) / 60
print("This took " + str(round(end , 2)) + " minutes to complete.")

