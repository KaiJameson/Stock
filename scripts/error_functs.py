from environment import error_file
import sys
import traceback
import time

def error_handler(symbol, exception):
    f = open(error_file, "a")
    f.write("Problem encountered with stock: " + symbol + "\n")
    f.write("Error is of type: " + str(type(exception)) + "\n")
    exit_info = sys.exc_info()
    f.write(str(exit_info[1]) + "\n")
    traceback.print_tb(tb=exit_info[2], file=f)
    f.close()
    print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

def net_error_handler(symbol, Exception):
    f = open(error_file, "a")
    f.write("\n\n EXCEPTION HANDLED \n\n")
    f.write("Error is of type: " + str(type(Exception)) + "\n")
    exit_info = sys.exc_info()
    f.write(str(exit_info[1]) + "\n")
    traceback.print_tb(tb=exit_info[2], file=f)
    f.close()
    time.sleep(1)
    pass

