import os
import sys 
from src.exception import CustomException
from src.logger import logging 
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import data_transform   
from src.components.data_transformation import data_trans_config


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifact',"train.csv")
    test_data_path:str = os.path.join('artifact',"test.csv")
    raw_data_path:str = os.path.join('artifact',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            df=pd.read_csv('data_container\stud.csv') 
            logging.info("read the dataset")
            ## creating the location where the raw data csv is to be stored ,where ingestion config has the sample path , and we are using that in makedir to create that directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)  
            ## after creating the directory , we are saving the file in csv , in the specified location
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test splitter initiated")
            train_set,test_set= train_test_split(df,test_size=0.3,random_state=43)
            
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
        
            logging.info("we have created the paths to all the datasets /ingestion of data is completed") 
            
            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e,sys)   
        
        
if __name__=="__main__":
    obj=DataIngestion()
    _,train_data,test_data=obj.initiate_data_ingestion() 
    data_transformation=data_transform()
    data_transformation.initiate_data_transformation(train_data,test_data)
    