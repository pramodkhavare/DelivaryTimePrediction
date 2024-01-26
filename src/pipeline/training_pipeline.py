from src.constant import * 
from src.config.configuration import *

from src.utils import save_obj
from src.logger import logging 
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.data_ingestion import DataIngestion

import os ,sys 
import pandas as pd 
import numpy as np 

class Train():
    def __init__(self):
        self.c = 0
        print(f"********************{self.c}*******************")
        
        
   
obj = DataIngestion()
train_data ,test_data =obj.initiate_data_ingestion()
data_transformer =DataTransformation()
train_arr ,test_arr ,_ =data_transformer.initiate_data_transformation(train_data ,test_data)
model_training = ModelTrainer()
model_training.initiate_model_training(train_arr ,test_arr)
    