from paca_model import nn_train_save
from config.symbols import load_save_symbols
from config.environ import defaults
from paca_model import configure_gpu
from functions.functions import check_directories
from functions.error_functs import error_handler, keyboard_interrupt
from functions.data_load_functs import load_all_data, get_proper_df
import psutil
import sys
import time






def pause_running_training():
    s = time.time()
    processes = {p.pid: p.info for p in psutil.process_iter(["name"])}
    python_processes_pids = []
    pause_list = []

    for process in processes:
        if processes[process]["name"] == "python.exe":
            python_processes_pids.append(process)

    for pid in python_processes_pids:
        if any("batch" in string for string in psutil.Process(pid).cmdline()):
            pause_list.append(pid)
        elif any("tuner" in string for string in psutil.Process(pid).cmdline()):
            pause_list.append(pid)

    for pid in pause_list:
        psutil.Process(pid).suspend()

    print(f"Pausing python files took {time.time() - s}")
    return pause_list

def resume_running_training(pause_list):
    for pid in pause_list:
        psutil.Process(pid).resume()

def save_models(symbols):
    configure_gpu()
    for symbol in symbols:
        try:
            df = get_proper_df(symbol, 0, "V2")
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
if __name__ == "__main__":
    check_directories()        
    s = time.perf_counter()
    pause_list = pause_running_training()
    save_models(load_save_symbols)
    resume_running_training(pause_list)
    end = (time.perf_counter() - s) / 60
    print("This took " + str(round(end , 2)) + " minutes to complete.")

