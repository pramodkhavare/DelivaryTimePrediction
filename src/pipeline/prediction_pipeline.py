from src.constant import * 
from src.config.configuration import *
from src.utils import save_obj , load_model
from src.logger import logging 
from src.exception import CustomException

import os ,sys 



class CustomData():
    def __init__(self,
                 Delivery_person_Age: float,
                 Delivery_person_Ratings: float,
                 Vehicle_condition: float,
                 multiple_deliveries: float,
                 Distance: float ,
                 Type_of_order :str,
                 Type_of_vehicle :str,
                Festival :str,
                City :str,
                Weather_conditions :str,
                Road_traffic_density :str
                 ):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings =Delivery_person_Ratings
        self.Vehicle_condition =Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.Distance = Distance 
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        
    def get_data_as_dataframe(self):
        try:
            custome_data_input_dict = {
                'Delivery_person_Age' : [self.Delivery_person_Age],
                'Delivery_person_Ratings' : [self.Delivery_person_Ratings],
                'Vehicle_condition' : [self.Vehicle_condition],
                'multiple_deliveries' : [self.multiple_deliveries],
                'Distance' : [self.Distance],
                'Type_of_order' : [self.Type_of_order],
                'Type_of_vehicle' : [self.Type_of_vehicle],
                'Festival' : [self.Festival],
                'City' : [self.City],
                'Weather_conditions' : [self.Weather_conditions],
                'Road_traffic_density' : [self.Road_traffic_density]
            }
            
            df = pd.DataFrame(custome_data_input_dict)
            return  df 
        except Exception as e:
            logging.info("Eror in input data")
            raise CustomException(e ,sys )
        
        



class PredictionPipeline():
    def __init__(self):
        pass 
    def predict(self ,features):
        try:
            logging.info('Prediction started')
            preprocessor_file_path =PREPROCESSOR_OBJ_FILE_PATH 
            model_file_path  = MODEL_TRAINER_FILE_PATH
        
            preprocessor = load_model(preprocessor_file_path)
            model = load_model(model_file_path)
        
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data) 
            logging.info('Successfully predicted output ')
            return pred
        
        
        except Exception  as e:
            logging.info('Error occured in prediction file while predicting output')
            raise CustomException(e ,sys)
        
    
