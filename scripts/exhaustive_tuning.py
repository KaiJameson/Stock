from alpaca_neural_net import make_neural_net
import os
import sys
import subprocess
import time
from functions import check_directories
from environment import config_directory, tuning_status_file, error_file
import traceback
check_directories()
start_time = time.time()
ticker = 'NRZ'

EPOCHS = 2000
UNITS = [256, 448, 768]
N_STEPS = [50, 100, 150, 200, 250, 300]
DROPOUT = [.3, .35, .4]


best_acc = 0
best_step = N_STEPS[0]
best_unit = UNITS[0]
best_drop = DROPOUT[0]

tuning_status_file = ticker + '.txt'
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
                perc, acc = make_neural_net(ticker, N_STEPS=step, UNITS=unit, DROPOUT=drop, EPOCHS=EPOCHS)
                e = time.time()
                m = (e - s) / 60
                f = open(tuning_status_file, 'a')
                f.write("Finished another run.\n")
                f.write("This run took " + str(m) + ' minutes to run.\n')
                f.write('\tUNITS:'+str(unit)+'\n')
                f.write('\tDROPOUT:'+str(drop)+'\n')
                f.write('\tN_STEPS:'+str(step)+'\n')
                f.write('\tEPOCHS:' + str(EPOCHS) + '\n')
                f.write('The accuracy for this run is ' + str(acc) + '\n')
                f.close()
                if acc > best_acc:
                    best_acc = acc
                    best_step = step
                    best_unit = unit
                    best_drop = drop
            except:
                f = open(error_file, 'a')
                f.write('Problem with running exhaustive tuning on these settings for ticker ' + ticker + '\n')
                f.write('\tUNITS:'+str(unit)+'\n')
                f.write('\tDROPOUT:'+str(drop)+'\n')
                f.write('\tN_STEPS:'+str(step)+'\n')
                f.write('\tEPOCHS:' + str(EPOCHS) + '\n')
                exit_info = sys.exc_info()
                f.write(str(exit_info[1]) + '\n')
                traceback.print_tb(tb=exit_info[2], file=f)
                f.close()
                print('\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n')
                
config_file = config_directory + '/' + ticker + '.csv'
f = open(config_file, 'w')
f.write('UNITS,'+str(best_unit)+'\n')
f.write('DROPOUT,'+str(best_drop)+'\n')
f.write('N_STEPS,'+str(best_step)+'\n')
f.write('EPOCHS,' + str(EPOCHS) + '\n')
f.close()
f = open(tuning_status_file, 'a')
f.write('\nDone with ' + ticker + '\n')
end_time = time.time()
total_time = end_time - start_time
total_minutes = total_time / 60
f.write('It took ' + str(total_minutes) + ' minutes to complete.\n')

print("The bests are:")
print('UNITS: '+str(best_unit)+'\n')
print('DROPOUT: '+str(best_drop)+'\n')
print('N_STEPS: '+str(best_step)+'\n')
print('EPOCHS:' + str(EPOCHS) + '\n')
print("and it took " + str(total_minutes) + " minutes to complete.\n")

f.close()



