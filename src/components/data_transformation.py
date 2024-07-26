from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass 
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sys 
from src.logger import logging
from src.exception import CustomException
import os
from src.utils import save_object
import pickle

@dataclass
class data_trans_config:
    preprocessor_pb_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class data_transform:
    def __init__(self):
        self.data_transform_config = data_trans_config()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("OHE", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # Update to avoid centering sparse matrices
                ]
            )

            logging.info(f"categorical_features: {categorical_features}")
            logging.info(f"numerical_features: {numerical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the files")
            logging.info("Now getting the preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_feature = "math_score"
            numerical_features = ['reading_score', 'writing_score']

            input_train_data = train_df.drop(columns=[target_feature], axis=1)
            target_train_data = train_df[target_feature]

            input_test_data = test_df.drop(columns=[target_feature], axis=1)
            target_test_data = test_df[target_feature]

            logging.info('Applying preprocessing on all the data (training and testing)')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_test_data)

            train_arr = np.c_[input_feature_train_arr, np.array(target_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_data)]

            logging.info("Saved the preprocessing steps")

            save_object(
                file_path=self.data_transform_config.preprocessor_pb_file_path,
                obj=preprocessing_obj
            )

            return {
                "train_arr": train_arr,
                "test_arr": test_arr,
                "preprocessor_file_path": self.data_transform_config.preprocessor_pb_file_path,
            }
        except Exception as e:
            raise CustomException(e, sys)
