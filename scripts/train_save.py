from alpaca_neural_net import saveload_neural_net
from symbols import load_save_symbols
from functions import check_directories
from environment import model_saveload_directory, error_file, config_directory
import traceback
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
                saveload_neural_net(symbol, UNITS=int(values["UNITS"]), DROPOUT=float(values["DROPOUT"]),
                N_STEPS=int(values["N_STEPS"]), EPOCHS=int(values["EPOCHS"]), SAVELOAD=True)
            else:
                saveload_neural_net(symbol, EPOCHS=5000, PATIENCE=800, SAVELOAD=True)

        except KeyboardInterrupt:
            print("I acknowledge that you want this to stop.")
            print("Thy will be done.")
            sys.exit(-1)
        except:
            f = open(error_file, "a")
            f.write("problem with configged stock: " + symbol + "\n")
            exit_info = sys.exc_info()
            f.write(str(exit_info[1]) + "\n")
            traceback.print_tb(tb=exit_info[2], file=f)
            f.close()
            print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")
        
s = time.time()
save_models(load_save_symbols)
end = time.time() - s
print("This took " + str(s) + "minutes to complete.")

