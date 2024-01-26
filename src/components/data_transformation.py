from src.constant import * 
from src.logger import logging 
from src.exception import CustomException

import os ,sys 
from src.config.configuration import *
from src.utils import *
from dataclasses import dataclass 
from sklearn.base import BaseEstimator ,TransformerMixin 
import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler ,OrdinalEncoder ,OneHotEncoder
import math

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



class Feature_Engineering(BaseEstimator ,TransformerMixin):
    def __init__(self):
        logging.info('********************Feature Engineering Started*******************')
        
        
    def get_distance(self ,lat1 ,long1 ,lat2 ,long2):
        
        lat1 ,long1 ,lat2 ,long2  = map(math.radians, [lat1 ,long1 ,lat2 ,long2])
    
        #Haversine Formula
        dlat =lat2 - lat1
        dlon =long2 - long1
    
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) *math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a) , math.sqrt(1-a))
        R = 6371.0  #Earth radius in km
    
        dist = R * c
    
        return dist
        
    def drop_column(self ,df):
        try:
            df.drop(columns=['ID','Delivery_person_ID', 'Restaurant_latitude' ,'Restaurant_longitude' ,
                              'Delivery_location_latitude' ,'Delivery_location_longitude' , 
                              'Order_Date' ,'Time_Orderd' ,'Time_Order_picked' ,
                              ] ,axis=1 ,inplace=True)
            
            return df
        
        except Exception as e:
            raise CustomException(e ,sys)
        

    def transform_data(self ,df):
        try:
            df['Distance'] = df.apply(lambda x : self.get_distance(x['Restaurant_latitude'] , 
                                                          x['Restaurant_longitude'] ,
                                                          x['Delivery_location_latitude'] 
                                                          ,x['Delivery_location_longitude']),axis=1)
            
            df = self.drop_column(df)
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def transform(self ,df):
        try:
            transformed_df = self.transform_data(df)
            
            return transformed_df
            
        except Exception as e:
            raise CustomException(e ,sys)
        
        

@dataclass 
class DataTransformationConfig():
    preprocessor_obj_file_path = PREPROCESSOR_OBJ_FILE_PATH 
    feature_eng_obj_file_path = FEATURE_ENGINEERING_OBJ_FILE_PATH
    transformed_train_file_path = TRANSFORMED_TRAIN_FILE_PATH 
    transformed_test_file_path = TRANSFORMED_TEST_FILE_PATH 
    

class DataTransformation():
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            logging.info('Transformation Pipeline Initiated')
            
            road_traffic = ['Low' ,'Medium' ,'High' ,'Jam']
            weather_cond = ['Sunny' ,'Cloudy' ,'Windy' ,'Fog' ,'Sandstorms' ,'Stormy'] 
            
            nominal_encodeing_col = ['Type_of_order','Type_of_vehicle','Festival','City',] 
            ordinal_encoding_col = ['Weather_conditions','Road_traffic_density']
            numerical_col = ['Delivery_person_Age','Delivery_person_Ratings',
                             'Vehicle_condition',
                             'multiple_deliveries','Distance']
            
            #Numerical_Pipeline
            numerical_pipeline =Pipeline(
                steps=[
                    ('impute' ,SimpleImputer(strategy='constant' ,fill_value=0)),
                    ('scalar' ,StandardScaler(with_mean=False))
                ]
            )
            
            #categorical data pipeline -Ordinal Pipeline
            Ordinal_pipeline = Pipeline(
                steps=[
                    ('impute' ,SimpleImputer(strategy='most_frequent')),
                    ('encoder' ,OrdinalEncoder(categories=[weather_cond ,road_traffic])),
                    ('scalar' ,StandardScaler(with_mean=False))
                ]
            )
            
            #Categorical data pipeline -Nominal Pipeline
            Nominal_pipeline = Pipeline(
                steps=[
                    ('impute' ,SimpleImputer(strategy='most_frequent')),
                    ('encoder' ,OneHotEncoder(handle_unknown='ignore')),
                    ('scalar' ,StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor =ColumnTransformer(
                [
                    ('numerical_pipeline' ,numerical_pipeline ,numerical_col),
                    ('Ordinal_pipeline' ,Ordinal_pipeline ,ordinal_encoding_col),
                    ('nominal pipeline' , Nominal_pipeline ,nominal_encodeing_col)
                ]
            )
            
            logging.info('Pipeline completed')
            return preprocessor 
        except Exception as e:
            logging.info('Unable to create transformation pipeline')
            raise CustomException(e ,sys)
        
    def get_feature_engineering_obj(self):
        try:
            feature_engineering_obj = Pipeline(
                steps=[
                    ('FE' ,Feature_Engineering())
                ]
            )
            return feature_engineering_obj
    
        except Exception as e:
            logging.info('Unable to create feature engineering pipeline')
            raise CustomException(e ,sys) 
    
    def initiate_data_transformation(self ,train_path ,test_path):
        try:
            logging.info('Initiated Data Transformation')
        
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            
            logging.info('Transforming data')
            
            fe_object =self.get_feature_engineering_obj()
            train_df = fe_object.transform(train_df)
            test_df = fe_object.transform(test_df)
            
            
        
            preprocessor_obj = self.get_data_transformation_obj()
            
            target_column = "Time_taken (min)"
            
            X_train = train_df.drop(columns = target_column ,axis=1)
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns = target_column ,axis =1)
            y_test = test_df[target_column]
            
            
            
            X_train = preprocessor_obj.fit_transform(X_train)
            X_test = preprocessor_obj.transform(X_test)
        
            train_arr = np.c_[X_train , np.array(y_train)]
            test_arr = np.c_[X_test ,np.array(y_test)]
            
            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)
            
            
            os.makedirs(os.path.dirname(self.datatransformationconfig.transformed_train_file_path) ,exist_ok=True)
            df_train.to_csv(self.datatransformationconfig.transformed_train_file_path ,index = False ,header = True)
            os.makedirs(os.path.dirname(self.datatransformationconfig.transformed_test_file_path) ,exist_ok=True)
            df_test.to_csv(self.datatransformationconfig.transformed_test_file_path  ,index = False ,header = True)
            
            
            
            
            save_obj(
                file_path=self.datatransformationconfig.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            save_obj(
                file_path=self.datatransformationconfig.feature_eng_obj_file_path ,
                obj= fe_object
            )
            
            
            logging.info('We are able to transform data successfully')
            
            return (
                train_arr,
                test_arr,
                self.datatransformationconfig.preprocessor_obj_file_path
            )
            
            
            
        except Exception as e:
            logging.info('Unable to transform data')
            raise CustomException(e ,sys)
    
            
        