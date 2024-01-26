from src.constant import * 
from src.config.configuration import *
from src.utils import save_obj

from src.logger import logging 
from src.exception import CustomException
from src.utils import evaluate_model


import os ,sys 
from dataclasses import dataclass  
import numpy as np 
import pandas as pd 
import math



from sklearn.tree import DecisionTreeRegressor 
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor 
from xgboost import XGBRegressor


@dataclass
class ModelTrainingConfig():
    model_trainer_file_path = MODEL_TRAINER_FILE_PATH
    
    
class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        
    def initiate_model_training(self ,train_arr ,test_arr):        
        try:
            logging.info('Model training started')
            
            #spliting data  into dependent and independet variable
            X_train ,y_train ,X_test ,y_test = (train_arr[:,:-1] ,train_arr[:,-1],
                                                test_arr[: ,:-1] ,test_arr[: ,-1])
            
            
            models ={
                'Random Forest' : RandomForestRegressor(),
                'DecisionTreeRegressor' : DecisionTreeRegressor(),
                'SVR' : SVR(),
                'GradientBoostRegressor' : GradientBoostingRegressor(),
                'XGBRegressor' : XGBRegressor()
            }
            
            
            model_report:dict = evaluate_model(X_train ,y_train ,X_test ,y_test ,models)
            
            print(model_report)
            
            logging.info(model_report)
            
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            logging.info(best_model_name)
            
            best_model = models[best_model_name] 
            
            logging.info(f"Best model is {best_model} , R2 score is {best_model_score}")
            
            print("********************************************************************************")
            
            print(f"Best model is {best_model} , R2 score is {best_model_score}")
            
            print("********************************************************************************")


            
            save_obj(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )
            
            logging.info('We are able to train model successfully')
            
            
        
        
        except Exception as e:
            logging.info("Unable to train model")
            raise CustomException(e ,sys)
        
    