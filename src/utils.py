import os 
import sys 
import numpy as np 
import pandas as pd 
from src.exception import CustomException
import dill
import pickle

def save_object(file_path, obj):
    try:
        ## this is how we get the directory name
        dir_path = os.path.dirname(file_path)
        ## creating the complete directory 
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    