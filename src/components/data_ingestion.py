import os ,sys 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass 
from src.logger import logging 
from src.exception import CustomException 
from src.config.configuration import *
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer



@dataclass
class DataIngestionConfig():
    raw_file_path = RAW_FILE_PATH 
    raw_train_file_path =RAW_TRAIN_FILE_PATH 
    raw_test_file_path = RAW_TEST_FILE_PATH 

class DataIngestion():
    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion started')
            df = pd.read_csv(INITIAL_DATASET_PATH)
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_file_path) ,exist_ok=True)
            df.to_csv(self.dataingestionconfig.raw_file_path)
            
             
            logging.info('Splitting Data into train and test data')
            train_data ,test_data = train_test_split(df ,test_size=0.2 ,random_state=42)
            
            
            logging.info('Adding training data into csv file')
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_train_file_path) ,exist_ok=True)
            df.to_csv(self.dataingestionconfig.raw_train_file_path)
            
            logging.info('Adding testing data into csv file')
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_test_file_path) ,exist_ok=True)
            df.to_csv(self.dataingestionconfig.raw_test_file_path)
            
            logging.info('Data ingestion completed')
            
            return(
                self.dataingestionconfig.raw_train_file_path ,
                self.dataingestionconfig.raw_test_file_path
            )
            
        except Exception as e:
            logging.info('Unable to ingest data')
            raise CustomException(e ,sys)
        
        
        
# # # Data Ingestion 
# if __name__ == '__main__':
#     obj = DataIngestion()
#     train_data_path ,test_data_path =obj.initiate_data_ingestion()


#Data Transformation
# if __name__ == '__main__':
#     obj = DataIngestion()
#     train_data_path ,test_data_path =obj.initiate_data_ingestion()
#     data_transformer =DataTransformation()
#     train_arr ,test_arr ,_ =data_transformer.initiate_data_transformation(train_data_path ,test_data_path)

# Model _training
if __name__ == '__main__':
    obj = DataIngestion()
    train_data ,test_data =obj.initiate_data_ingestion()
    data_transformer =DataTransformation()
    train_arr ,test_arr ,_ =data_transformer.initiate_data_transformation(train_data ,test_data)
    model_training = ModelTrainer()
    model_training.initiate_model_training(train_arr ,test_arr)

    
