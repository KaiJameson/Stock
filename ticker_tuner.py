import os
import sys
import subprocess

args = sys.argv()
if len(args)<2:
    print('You need to give me a ticker to test')
    sys.exit(-1)
ticker = args[2].upper()
param_names = ['UNITS', 'DROPOUT', 'N_STEPS', 'EPOCHS']
for param in param_names:
    print('calling tune.py for ' + param)
    subprocess.call(['tune.py', param, ticker])

