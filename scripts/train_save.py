from alpaca_neural_net import saveload_neural_net
from symbols import load_save_symbols
from functions import check_directories, error_handler
from environment import model_saveload_directory, error_file, config_directory, defaults
import os
import sys
import time

check_directories()

def save_models(symbols):
    for symbol in symbols:
        try:
            config_name = config_directory + "/" + symbol + ".csv"
            if os.path.isfile(config_name):
                f = open(config_name, "r")
                values = {}
                for line in f:
                    parts = line.strip().split(",")
                    values[parts[0]] = parts[1]
                epochs = saveload_neural_net(symbol, UNITS=int(values["UNITS"]), DROPOUT=float(values["DROPOUT"]),
                N_STEPS=int(values["N_STEPS"]), EPOCHS=int(values["EPOCHS"]), SAVELOAD=True)
            else:
                epochs = saveload_neural_net(symbol, SAVELOAD=True)

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

