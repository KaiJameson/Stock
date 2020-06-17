from alpaca_neural_net import tuning_neural_net
from symbols import exhaustive_symbols
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
UNITS = [256, 448, 768]
N_STEPS = [50, 100, 150, 200, 250, 300]
DROPOUT = [.3, .35, .4]

done_dir = tuning_directory + '/done'
if not os.path.isdir(done_dir):
    os.mkdir(done_dir)
current_dir = tuning_directory + '/current'
if not os.path.isdir(current_dir):
    os.mkdir(current_dir)
f_name = done_dir + '/' + ticker + '.txt'
file_name = current_dir + '/' + ticker + '.csv'
tuning_status_file = tuning_directory + '/' + ticker + '.txt'
if not os.path.isfile(tuning_status_file):
    f = open(tuning_status_file, 'w')
    f.close()
done_message = 'You are done tuning this stock.'
def get_info():
    if not os.path.isfile(f_name):
        if os.path.isfile(file_name):
            '''
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
            '''
            f = open(file_name, 'r')
            info = []
            for line in f:
                if line.startswith('f'):
                    info.append(float(line.strip('f').strip()))
                elif line.startswith('i'):
                    info.append(int(line.strip('i').strip()))
                else:
                    info.append(line.strip())
            if len(info) != 9:
                #TODO: RAISE ERROR
                print('YOU NEED TO RAISE AN ERROR HERE')
            print('INFO:', info)
            return info
        else:
            best_mae = 0
            best_step = N_STEPS[0]
            best_unit = UNITS[0]
            best_drop = DROPOUT[0]
            info = (2 * [best_mae, best_step, best_unit, best_drop]) + [get_end_date()]
            info[4] = 0
            print('NEW INFO:', info)
            return info
    else:
        #delete the config file and move on
        if os.path.isfile(file_name):
            os.remove(file_name)
        return ''

def write_info(info, total_time=0):
    '''
    best mae
    best n step
    best unit
    best drop
    total time
    recent n_step
    recent unit
    recent dropout
    end_date
    '''
    if info[5] == N_STEPS[-1] and info[6] == UNITS[-1] and info[7] == DROPOUT[-1]:
        config_file = config_directory + '/' + ticker + '.csv'
        print_params(config_file, info[1], info[2], info[3], EPOCHS, punct=',')
        if os.path.isfile(file_name):
            os.remove(file_name)
        f = open(tuning_status_file, 'a')
        f.write('\nDone with ' + ticker + '\n')
        total_minutes = info[4] / 60
        time_message = 'It took ' + str(total_minutes) + ' minutes to complete.\n'
        f.write(time_message)
        f.close()
        f = open(f_name, 'w')
        f.write('Done with this ticker\n')
        f.write(time_message)
        f.write('The best mean absolute error was ' + str(info[0]) + '\n')
        f.close()
        print_params(f_name, info[1], info[2], info[3],  EPOCHS, punct=': ')
        print('THIS FILE IS DONE TUNING')
    else:
        f = open(file_name, 'w')
        info[4] += total_time
        print('WRITING TO FILE NAME WITH INFO:', info)
        for num in info:
            if isinstance(num, float):
                f.write('f'+str(num)+'\n')
            elif isinstance(num, int):
                f.write('i'+str(num)+'\n')
            elif isinstance(num, str):
                f.write(num + '\n')
            else:
                f.close()
                #TODO:raise error
                sys.exit('THERE WAS A PROBLEM WITH YOUR INFO ARRAY')
        f.close()
        return total_time

def print_params(file_name, step, unit, drop, epoch, indent='', punct=','):
    f = open(file_name, 'a')
    f.write(indent + 'N_STEPS' + punct + str(step) + '\n')
    f.write(indent + 'UNITS' + punct + str(unit) + '\n')
    f.write(indent + 'DROPOUT' + punct + str(drop) + '\n')
    f.write(indent + 'EPOCHS' + punct + str(epoch) + '\n')
    f.close()

check_directories()
start_time = time.time()

def get_end_date():
    tz = 'US/EASTERN'
    now = time.time()
    n = datetime.datetime.fromtimestamp(now)
    date = n.date()
    year = date.year
    month = date.month
    day = date.day
    end_date = datetime.datetime(year, month, day)
    end_date = time.mktime(end_date.timetuple())
    end_date = pd.Timestamp(end_date, unit='s', tz=tz).isoformat()
    return end_date


def get_indexs(info):
    print('GETTING INDEXS FOR:', info)
    if info[0] == 0:
        print('0,0,0')
        return 0, 0, 0
    drop_index = DROPOUT.index(info[7])
    print('DROP_INDEX:', drop_index)
    drop_index = (drop_index + 1) % len(DROPOUT)
    print('DROP_INDEX:', drop_index)
    unit_index = UNITS.index(info[6])
    start_unit = unit_index
    print('UNIT_INDEX:', unit_index)
    if drop_index == 0:
        print('INCREMENTING UNIT')
        unit_index = (unit_index + 1) % len(UNITS)
        print('UNIT_INDEX:', unit_index)
    step_index = N_STEPS.index(info[5])
    start_step = step_index
    print('STEP_INDEX:', step_index)
    if unit_index == 0 and start_unit != 0:
        print('INCREMENTING STEP')
        step_index = (step_index + 1) % len(N_STEPS)
        print('STEP_INDEX:', step_index)
    if step_index == 0 and start_step != 0:
        write_info(info)
        print('SHOULD NOT HAVE MADE IT HERE')
        sys.exit(-1)
    return drop_index, unit_index, step_index


done = False
while not done:
    try:
        print('\nStarting a new iteration\n')
        info = get_info()
        if info == '':
            print(done_message)
            done = True
            break
        drop_index, unit_index, step_index = get_indexs(info)
        start_time = time.time()
        drop = DROPOUT[drop_index]
        info[7] = drop
        unit = UNITS[unit_index]
        info[6] = unit
        step = N_STEPS[step_index]
        info[5] = step

        acc, mae = tuning_neural_net(
            ticker, 
            end_date=info[8], 
            N_STEPS=step, 
            UNITS=unit, 
            DROPOUT=drop, 
            EPOCHS=EPOCHS
        )

        end_time = time.time()
        total_time = end_time - start_time
        if mae < info[0]:
            #mae, step, unit, drop
            info[0] = mae
            info[1] = step
            info[2] = unit
            info[3] = drop
        elif info[0] == 0:
            info[0] = mae
            info[1] = step
            info[2] = unit
            info[3] = drop

        m = total_time / 60
        f = open(tuning_status_file, 'a')
        f.write("Finished another run.\n")
        f.write("This run took " + str(m) + ' minutes to run.\n')
        f.close()
        print_params(tuning_status_file, step, unit, drop, EPOCHS, indent='\t', punct=': ')
        f = open(tuning_status_file, 'a')
        f.write('The mean absolute error for this run is: ' + str(round(mae, 4)) + '\n')
        f.write('The accuracy for this this run is: ' + str(round(acc * 100, 2)) + '\n')
        f.close()
        write_info(info, total_time=total_time)
    except KeyboardInterrupt:
        print('I acknowledge that you want this to stop')
        print('Thy will be done')
        sys.exit(-1)
    except:
        exit_info = sys.exc_info()
        f = open(error_file, 'a')
        f.write('Problem with running exhaustive tuning on these settings for ticker: ' + ticker + '\n')
        f.close()
        print_params(error_file, step, unit, drop, EPOCHS, indent='\t', punct=': ')
        f = open(error_file, 'a')
        f.write(str(exit_info[1]) + '\n')
        traceback.print_tb(tb=exit_info[2], file=f)
        f.close()
        print('\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n')
        sys.exit('CHECK THE ERROR FILE')

