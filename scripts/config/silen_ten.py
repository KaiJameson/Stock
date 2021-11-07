import os
import logging

def silence_tensorflow():    
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
