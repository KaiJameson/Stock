import os
import logging


quiet_tensorflow = True

def silence_tensorflow():    
    if quiet_tensorflow:
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    else:
        print("So you have chosen ..... death")
        pass
