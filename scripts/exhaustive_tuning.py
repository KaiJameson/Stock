from alpaca_neural_net import tuning_neural_net
import os
import sys
import subprocess
import time
from functions import check_directories
from environment import config_directory, tuning_directory, error_file
import traceback
import datetime
import pandas as pd



def print_params(file_name, unit, drop, step, epoch, indent='', punct=','):
    f = open(file_name, 'a')
    f.write(indent + 'UNITS' + punct + str(unit) + '\n')
    f.write(indent + 'DROPOUT' + punct + str(drop) + '\n')
    f.write(indent + 'N_STEPS' + punct + str(step) + '\n')
    f.write(indent + 'EPOCHS' + punct + str(epoch) + '\n')
    f.close()

check_directories()
start_time = time.time()
ticker = 'TGI'


EPOCHS = 2000
UNITS = [256, 448, 768]
N_STEPS = [50, 100, 150, 200, 250, 300]
DROPOUT = [.3, .35, .4]

tz = 'US/EASTERN'
now = time.time()
n = datetime.datetime.fromtimestamp(now)
date = n.date()
year = date.year
month = date.month
day = date.day
end_date = datetime.datetime(year, month, day-1)
end_date = time.mktime(end_date.timetuple())
end_date = pd.Timestamp(end_date, unit='s', tz=tz).isoformat()


best_acc = 0
best_step = N_STEPS[0]
best_unit = UNITS[0]
best_drop = DROPOUT[0]
done_dir = tuning_directory + '/done'
if not os.path.isdir(done_dir):
    os.mkdir(done_dir)
tuning_status_file = tuning_directory + '/' + ticker + '.txt'
if not os.path.isfile(tuning_status_file):
    f = open(tuning_status_file, 'w')
    f.close()
f = open(tuning_status_file, 'a')
f.write("\n\nstarting to tune " + ticker + '\n\n')
f.close()
for step in N_STEPS:
    for unit in UNITS:
        for drop in DROPOUT:
            try:
                s = time.time()
                acc = tuning_neural_net(ticker, end_date, N_STEPS=step, UNITS=unit, DROPOUT=drop, EPOCHS=EPOCHS)
                e = time.time()
                m = (e - s) / 60
                f = open(tuning_status_file, 'a')
                f.write("Finished another run.\n")
                f.write("This run took " + str(m) + ' minutes to run.\n')
                f.close()
                print_params(tuning_status_file, unit, drop, step, EPOCHS, indent='\t', punct=': ')
                f = open(tuning_status_file, 'a')
                f.write('The accuracy for this run is ' + str(acc) + '\n')
                f.close()
                if acc > best_acc:
                    best_acc = acc
                    best_step = step
                    best_unit = unit
                    best_drop = drop
            except KeyboardInterrupt:
                print('I acknowledge that you want this to stop')
                print('Thy will be done')
                sys.exit(-1)
            except:
                f = open(error_file, 'a')
                f.write('Problem with running exhaustive tuning on these settings for ticker ' + ticker + '\n')
                f.close()
                print_params(error_file, unit, drop, step, EPOCHS, indent='\t', punct=': ')
                exit_info = sys.exc_info()
                f = open(error_file, 'a')
                f.write(str(exit_info[1]) + '\n')
                traceback.print_tb(tb=exit_info[2], file=f)
                f.close()
                print('\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n')
                
config_file = config_directory + '/' + ticker + '.csv'
print_params(config_file, ',', best_unit, best_drop, best_step, EPOCHS)
f = open(tuning_status_file, 'a')
f.write('\nDone with ' + ticker + '\n')
end_time = time.time()
total_time = end_time - start_time
total_minutes = total_time / 60
time_message = 'It took ' + str(total_minutes) + ' minutes to complete.\n'
f.write(time_message)
f.close()

print("The bests are:")
print('UNITS: '+str(best_unit)+'\n')
print('DROPOUT: '+str(best_drop)+'\n')
print('N_STEPS: '+str(best_step)+'\n')
print('EPOCHS:' + str(EPOCHS) + '\n')
print(time_message)
