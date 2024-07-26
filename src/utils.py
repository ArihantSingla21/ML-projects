import os 
import sys 
import numpy as np 
import pandas as pd 
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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


def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}
        
        for i in range(len(models)):
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model= list(models.values())[i]
            
            model.fit(x_train,y_train)
            
            y_train_pred= model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_model_r2s= r2_score(y_train,y_train_pred)
            test_model_r2s= r2_score(y_test,y_test_pred)
            
            report[list(model.keys())[i]]= test_model_r2s    
        return report
            
    except Exception as e:
        raise CustomException(e,sys)        

def load_object(file_path):
    try:
        with open(file_path,"rb") as obj:
            return dill.load(obj)       
    except Exception as e:
        raise CustomException(e,sys)    