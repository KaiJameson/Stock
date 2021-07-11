from environ import error_file
import sys
import traceback
import time
import datetime

def error_handler(symbol, exception):
    err_file = open(error_file, "a")
    write_exception_details(err_file, symbol, exception)
    write_exc_info(err_file)
    err_file.close()
    print("\nERROR ENCOUNTERED!! CHECK ERROR FILE!!\n")

def net_error_handler(symbol, exception):
    err_file = open(error_file, "a")
    err_file.write("\n\n EXCEPTION HANDLED \n\n")
    write_exception_details(err_file, symbol, exception)
    write_exc_info(err_file)
    err_file.close()
    print("EXCEPTION HANDLED", flush=True)
    time.sleep(3)
    
def write_exc_info(err_file):
    exit_info = sys.exc_info()
    err_file.write(str(exit_info[1]) + "\n")
    traceback.print_tb(tb=exit_info[2], file=err_file)

def write_exception_details(err_file, symbol, exception):
    err_file.write("Problem encountered with stock: " + symbol + "\n")
    err_file.write("Error is of type: " + str(type(exception)) + "\n")
    err_file.write("Error happened at: " + str(datetime.datetime.now()) + "\n")
