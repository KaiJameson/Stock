from alpaca_neural_net import tuning_neural_net
from symbols import exhaustive_symbols, exhaust_year, exhaust_month, exhaust_day
import os
import sys
import subprocess
import time
from functions import check_directories
from environment import config_directory, tuning_directory, error_file
import traceback
import datetime
import pandas as pd

ticker = exhaustive_symbols

check_directories()

EPOCHS = 2000
UNITS = [128, 256, 448]
N_STEPS = [150, 200, 250, 300, 350, 400, 450, 500]
DROPOUT = [.35, .4, .45]

iteration_num = len(UNITS) * len(N_STEPS) * len(DROPOUT)

done_dir = tuning_directory + "/done"
if not os.path.isdir(done_dir):
    os.mkdir(done_dir)
current_dir = tuning_directory + "/current"
if not os.path.isdir(current_dir):
    os.mkdir(current_dir)
f_name = done_dir + "/" + ticker + ".txt"
file_name = current_dir + "/" + ticker + ".csv"
tuning_status_file = tuning_directory + "/" + ticker + ".txt"
if not os.path.isfile(tuning_status_file):
    f = open(tuning_status_file, "w")
    f.close()
done_message = "You are done tuning this stock."

def get_info():
    if not os.path.isfile(f_name):
        if os.path.isfile(file_name):
            """
            Structure of this file will be
            best mae (f0.22)
            best n_step (50)
            best unit (448)
            best dropout (f0.3)
            total time
            recent n_step (300)
            recent unit (448)
            recent dropout (f0.4)
            end_date
            total test accuracy
            total validation accuracy
            total train accuracy
            total test mae
            total validation mae
            total train mae
            """
            f = open(file_name, "r")
            info = []
            for line in f:
                if line.startswith("f"):
                    info.append(float(line.strip("f").strip()))
                elif line.startswith("i"):
                    info.append(int(line.strip("i").strip()))
                else:
                    info.append(line.strip())
            if len(info) != 13:
                #TODO: RAISE ERROR
                print("YOU NEED TO RAISE AN ERROR HERE")
            print("INFO:", info)
            return info
        else:
            best_mae = 0
            best_step = N_STEPS[0]
            best_unit = UNITS[0]
            best_drop = DROPOUT[0]
            info = (2 * [best_mae, best_step, best_unit, best_drop]) + [get_end_date()]
            info[4] = 0
            info = info + [0, 0, 0, 0, 0, 0]
            print("NEW INFO:", info)
            return info
    else:
        #delete the config file and move on
        if os.path.isfile(file_name):
            os.remove(file_name)
        return ""

def write_info(info, total_time=0, test_acc=0, valid_acc=0, train_acc=0, test_mae=0,
valid_mae=0, train_mae=0):
    """
    best mae
    best n step
    best unit
    best drop
    total time
    recent n_step
    recent unit
    recent dropout
    end_date
    total test accuracy
    total validation accuracy
    total train accuracy
    total test mae
    total validation mae
    total train mae
    """
    if info[5] == N_STEPS[-1] and info[6] == UNITS[-1] and info[7] == DROPOUT[-1]:
        config_file = config_directory + "/" + ticker + ".csv"
        print_params(config_file, info[1], info[2], info[3], EPOCHS, punct=",")
        if os.path.isfile(file_name):
            os.remove(file_name)
        f = open(tuning_status_file, "a")
        f.write("\nDone with " + ticker + "\n")
        total_minutes = info[4] / 60
        time_message = "It took " + str(total_minutes) + " minutes to complete.\n"
        f.write(time_message)
        f.write("The average time was: " + str(round(total_minutes / iteration_num, 2)) + " minutes.\n")
        f.write("The average test accuracy was: " + str(round((info[9] / iteration_num) * 100, 2)) + "%\n")
        f.write("The average validation accuracy was: " + str(round((info[10] / iteration_num) * 100, 2)) + "%\n")
        f.write("The average train accuracy was: " + str(round((info[11] / iteration_num) * 100, 2)) + "%\n")
        f.write("The average test mae was: " + str(round(info[12] / iteration_num, 4)) + "\n")
        f.write("The average validation mae was: " + str(round(info[13] / iteration_num, 4)) + "\n")
        f.write("The average train mae was: " + str(round(info[14] / iteration_num, 4)) + "\n")
        f.close()

        f = open(f_name, "w")
        f.write("Done with this ticker\n")
        f.write(time_message)
        f.write("The best mean absolute error was " + str(info[0]) + "\n")
        
        f.close()
        print_params(f_name, info[1], info[2], info[3],  EPOCHS, punct=": ")
        print("THIS FILE IS DONE TUNING")
    else:
        f = open(file_name, "w")
        info[4] += total_time
        info[9] += test_acc
        info[10] += valid_acc
        info[11] += train_acc
        info[12] += test_mae
        info[13] += valid_mae
        info[14] += train_mae
        print("WRITING TO FILE NAME WITH INFO:", info)
        for num in info:
            if isinstance(num, float):
                f.write("f"+str(num)+"\n")
            elif isinstance(num, int):
                f.write("i"+str(num)+"\n")
            elif isinstance(num, str):
                f.write(num + "\n")
            else:
                f.close()
                #TODO:raise error
                sys.exit("THERE WAS A PROBLEM WITH YOUR INFO ARRAY")
        f.close()
        return total_time

def print_params(file_name, step, unit, drop, epoch, indent="", punct=","):
    f = open(file_name, "a")
    f.write(indent + "N_STEPS" + punct + str(step) + "\n")
    f.write(indent + "UNITS" + punct + str(unit) + "\n")
    f.write(indent + "DROPOUT" + punct + str(drop) + "\n")
    f.write(indent + "EPOCHS" + punct + str(epoch) + "\n")
    f.close()

check_directories()
start_time = time.time()

def get_end_date():
    tz = "US/EASTERN"
    end_date = datetime.datetime(exhaust_year, exhaust_month, exhaust_day)
    end_date = time.mktime(end_date.timetuple())
    end_date = pd.Timestamp(end_date, unit="s", tz=tz).isoformat()
    return end_date


def get_indexs(info):
    print("GETTING INDEXS FOR:", info)
    if info[0] == 0:
        print("0,0,0")
        return 0, 0, 0
    drop_index = DROPOUT.index(info[7])
    print("DROP_INDEX:", drop_index)
    drop_index = (drop_index + 1) % len(DROPOUT)
    print("DROP_INDEX:", drop_index)
    unit_index = UNITS.index(info[6])
    start_unit = unit_index
    print("UNIT_INDEX:", unit_index)
    if drop_index == 0:
        print("INCREMENTING UNIT")
        unit_index = (unit_index + 1) % len(UNITS)
        print("UNIT_INDEX:", unit_index)
    step_index = N_STEPS.index(info[5])
    start_step = step_index
    print("STEP_INDEX:", step_index)
    if unit_index == 0 and start_unit != 0:
        print("INCREMENTING STEP")
        step_index = (step_index + 1) % len(N_STEPS)
        print("STEP_INDEX:", step_index)
    if step_index == 0 and start_step != 0:
        write_info(info)
        print("SHOULD NOT HAVE MADE IT HERE")
        sys.exit(-1)
    return drop_index, unit_index, step_index


done = False
while not done:
    try:
        info = get_info()
        if info == "":
            print(done_message)
            done = True
            break

        print("\nStarting a new iteration:\n")
        drop_index, unit_index, step_index = get_indexs(info)
        start_time = time.time()
        drop = DROPOUT[drop_index]
        info[7] = drop
        unit = UNITS[unit_index]
        info[6] = unit
        step = N_STEPS[step_index]
        info[5] = step

        test_acc, valid_acc, train_acc, test_mae, valid_mae, train_mae = tuning_neural_net(
            ticker, 
            end_date=info[8], 
            N_STEPS=step, 
            UNITS=unit, 
            DROPOUT=drop, 
            EPOCHS=EPOCHS
        )

        end_time = time.time()
        total_time = end_time - start_time
        m = total_time / 60

        if test_mae < info[0]:
            #mae, step, unit, drop
            info[0] = test_mae
            info[1] = step
            info[2] = unit
            info[3] = drop
        elif info[0] == 0:
            info[0] = test_mae
            info[1] = step
            info[2] = unit
            info[3] = drop

        
        f = open(tuning_status_file, "a")
        f.write("Finished another run.\n")
        f.write("This run took " + str(m) + " minutes to run.\n")
        f.close()
        print_params(tuning_status_file, step, unit, drop, EPOCHS, indent="\t", punct=": ")
        f = open(tuning_status_file, "a")
        f.write("The test accuracy for this this run was: " + str(round(test_acc * 100, 2)) + "%\n")
        f.write("The validation accuracy for this this run was: " + str(round(valid_acc * 100, 2)) + "%\n")
        f.write("The train accuracy for this this run was: " + str(round(train_acc * 100, 2)) + "%\n")
        f.write("The test mean absolute error for this run was: " + str(round(test_mae, 4)) + "\n")
        f.write("The validation mean absolute error for this run was: " + str(round(valid_mae, 4)) + "\n")
        f.write("The train mean absolute error for this run was: " + str(round(train_mae, 4)) + "\n")
        f.close()
        write_info(info, total_time=total_time, test_acc=test_acc, valid_acc=valid_acc, train_acc=train_acc, 
        test_mae=test_mae, valid_mae=valid_mae, train_mae=train_mae)
    except KeyboardInterrupt:
        print("I acknowledge that you want this to stop.")
        print("Thy will be done.")
        sys.exit(-1)
    except:
        exit_info = sys.exc_info()
        f = open(error_file, "a")
        f.write("Problem with running exhaustive tuning on these settings for ticker: " + ticker + "\n")
        f.close()
        print_params(error_file, step, unit, drop, EPOCHS, indent="\t", punct=": ")
        f = open(error_file, "a")
        f.write(str(exit_info[1]) + "\n")
        traceback.print_tb(tb=exit_info[2], file=f)
        f.close()
        print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")
        sys.exit("CHECK THE ERROR FILE")

