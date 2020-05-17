from alpaca_neural_net import make_neural_net
import os

def find_best_stats(ticker):
    directory = 'config'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    unit = find_best_units(ticker)
    dropout = find_best_dropout(ticker)
    n_step = find_best_n_steps(ticker)
    epoch = find_best_epochs(ticker)
    f_name = directory + '/' + ticker + '.csv'
    f = open(f_name, 'w')
    f.write(ticker + ',' + unit + ',' + dropout + ',' + n_step + ',' + epoch)
    f.close()


def find_best_units(ticker):
    UNIT_VALS = [256, 320, 384, 448]
    best_unit = UNIT_VALS[0]
    best_acc = 0
    for unit in UNIT_VALS:
        perc, acc = make_neural_net(ticker, UNITS=unit)
        if acc > best_acc:
            best_acc = acc
            best_unit = unit
    return best_unit


def find_best_dropout(ticker):
    DROPOUTS = [.3, .325, .35, .375, .4, .425]
    best_drop = DROPOUTS[0]
    best_acc = 0
    for drop in DROPOUTS:
        perc, acc = make_neural_net(ticker, DROPOUT=drop)
        if acc > best_acc:
            best_acc = acc
            best_drop = drop
    return best_drop

def find_best_n_steps(ticker):
    N_STEP_VALS = [50, 100, 150, 200, 250, 300]
    best_step = N_STEP_VALS[0]
    best_acc = 0
    for step in N_STEP_VALS:
        perc, acc = make_neural_net(ticker, N_STEPS=step)
        if acc > best_acc:
            best_acc = acc
            best_step = step
    return best_step


def find_best_epochs(ticker):
    EPOCH_VALS = [500, 1000, 1500, 2000]
    best_epoch = EPOCH_VALS[0]
    best_acc = 0
    for epoch in EPOCH_VALS:
        perc, acc = make_neural_net(ticker, EPOCHS=epoch)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
    return best_epoch






