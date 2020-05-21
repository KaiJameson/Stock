from alpaca_neural_net import make_neural_net
import os
import sys
import subprocess
import time

start_time = time.time()
ticker = 'PENN'

EPOCHS = 2000
UNITS = [256, 448, 768]
N_STEPS = [50, 100, 150, 200, 250, 300]
DROPOUT = [.3, .35, .4]


best_acc = 0
best_step = N_STEPS[0]
best_unit = UNITS[0]
best_drop = DROPOUT[0]

status_file = 'status.txt'
if not os.path.isfile(status_file):
    f = open(status_file, 'w')
    f.close()
f = open(status_file, 'a')
f.write("\n\nstarting to tune " + ticker + '\n\n')
f.close()
for step in N_STEPS:
    for unit in UNITS:
        for drop in DROPOUT:
            s = time.time()
            perc, acc = make_neural_net(ticker, N_STEPS=step, UNITS=unit, DROPOUT=drop, EPOCHS=EPOCHS)
            e = time.time()
            m = (e - s) / 60
            f = open(status_file, 'a')
            f.write("finished another run\n")
            f.write("this run took " + str(m) + ' minutes to run\n')
            f.write('\tUNITS:'+str(unit)+'\n')
            f.write('\tDROPOUT:'+str(drop)+'\n')
            f.write('\tN_STEPS:'+str(step)+'\n')
            f.write('\tEPOCHS:2000\n')
            f.write('acc for this run is ' + str(acc) + '\n')
            f.close()
            if acc > best_acc:
                best_acc = acc
                best_step = step
                best_unit = unit
                best_drop = drop
config_file = 'config/' + ticker + '.csv'
f = open(config_file, 'w')
f.write('UNITS,'+str(best_unit)+'\n')
f.write('DROPOUT,'+str(best_drop)+'\n')
f.write('N_STEPS,'+str(best_step)+'\n')
f.write('EPOCHS,2000\n')
f.close()
f = open(status_file, 'a')
f.write('\nDone with ' + ticker + '\n')
end_time = time.time()
total_time = end_time - start_time
total_minutes = total_time / 60
f.write('it took ' + str(total_minutes) + ' minutes to complete\n')
f.close()



