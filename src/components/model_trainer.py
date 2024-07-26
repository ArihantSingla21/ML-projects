import pandas as pd 
import numpy as np
import os 
import sys
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainConfig():
    train_model_file_path = os.path.join("artifacts", "model.pickle")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()
        
    def model_train_initialize(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test data")
            x_train, x_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor()
            }
            
            # Ensure evaluate_model function returns a dictionary
            model_report = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
            
            logging.info("Model evaluation results obtained")
            
            # Check if model_report is a dictionary
            if not isinstance(model_report, dict):
                raise TypeError("evaluate_model did not return a dictionary")
            
            logging.info("Finding the best model")
            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise Exception("No suitable model found with a score above 0.6")
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
