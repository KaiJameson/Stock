from paca_model import nn_train_save
from config.symbols import load_save_symbols
from config.environ import defaults
from paca_model import configure_gpu
from functions.functions import check_directories
from functions.error_functs import error_handler, keyboard_interrupt
from functions.data_load_functs import load_all_data, get_proper_df
import sys
import time




check_directories()

def save_models(symbols):
    configure_gpu()
    for symbol in symbols:
        try:
            df = get_proper_df(symbol, 0, "training")
            data_dict = load_all_data(defaults, df)
            print(f"\n~~~ Now Training {symbol} ~~~")
            for predictor in defaults["ENSEMBLE"]:
                if "nn"in predictor:
                    print(f"\nTraining submodel {predictor} ...")
                    epochs = nn_train_save(symbol, end_date=None, params=defaults, predictor=predictor,
                        data_dict=data_dict[predictor])

        except KeyboardInterrupt:
            keyboard_interrupt()
        except Exception:
            error_handler(symbol, Exception)
        
s = time.perf_counter()
save_models(load_save_symbols)
end = (time.perf_counter() - s) / 60
print("This took " + str(round(end , 2)) + " minutes to complete.")

