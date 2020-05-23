from alpaca_neural_net import make_neural_net
import os
import sys
import subprocess
import time
from functions import check_directories
from environment import config_directory, tuning_status_file, error_file
check_directories()
start_time = time.time()
ticker = 'APDN'

EPOCHS = 2000
UNITS = [256, 448, 768]
N_STEPS = [50, 100, 150, 200, 250, 300]
DROPOUT = [.3, .35, .4]


best_acc = 0
best_step = N_STEPS[0]
best_unit = UNITS[0]
best_drop = DROPOUT[0]

tuning_status_file = 'status.txt'
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
                f.write("finished another run\n")
                f.write("this run took " + str(m) + ' minutes to run\n')
                f.write('\tUNITS:'+str(unit)+'\n')
                f.write('\tDROPOUT:'+str(drop)+'\n')
                f.write('\tN_STEPS:'+str(step)+'\n')
                f.write('\tEPOCHS:' + str(EPOCHS) + '\n')
                f.write('acc for this run is ' + str(acc) + '\n')
                f.close()
                if acc > best_acc:
                    best_acc = acc
                    best_step = step
                    best_unit = unit
                    best_drop = drop
            except:
                f = open(error_file, 'a')
                f.write('problem with running exhaustive tuning on these settings for ticker ' + ticker + '\n')
                f.write('\tUNITS:'+str(unit)+'\n')
                f.write('\tDROPOUT:'+str(drop)+'\n')
                f.write('\tN_STEPS:'+str(step)+'\n')
                f.write('\tEPOCHS:' + str(EPOCHS) + '\n')
                f.write(str(sys.exc_info()[1]) + '\n')
                f.close()
                
config_file = config_directory + '/' + ticker + '.csv'
f = open(config_file, 'w')
f.write('UNITS,'+str(best_unit)+'\n')
f.write('DROPOUT,'+str(best_drop)+'\n')
f.write('N_STEPS,'+str(best_step)+'\n')
f.write('EPOCHS,2000\n')
f.close()
f = open(tuning_status_file, 'a')
f.write('\nDone with ' + ticker + '\n')
end_time = time.time()
total_time = end_time - start_time
total_minutes = total_time / 60
f.write('it took ' + str(total_minutes) + ' minutes to complete\n')
f.close()



