import os ,sys
from datetime import datetime 
import pandas as pd 


def get_current_time_stamp():
    return(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")


ROOT_DIR = os.getcwd()

INITIAL_DATA_DIR = 'notebook\\data'

INITIAL_DATA_DATASET = 'finalTrain.csv'

ARTIFACT_DIR = 'Artifact'

DATA_INGESTION_DIR = "Data_ingestion"

CURRENT_TIME_STAMP = get_current_time_stamp()

DATA_INGESTION_INGESTED_DATA_DIR ='Ingested_Dir'

DATA_INGESTION_INGESTED_DATA_TEST_DATASET = 'raw_test.csv'

DATA_INGESTION_INGESTED_DATA_TRAIN_DATASET = 'raw_train.csv'


DATA_INGESTION_RAW_DATA_DIR = 'Raw_Dir'
DATA_INGESTION_RAW_DATA_DATASET = 'raw.csv'

DATA_TRANSFORMATION_DIR = 'Data_Transformation'

DATA_TRANSFORMATION_PROCESSOR_DIR = 'processor' 
DATA_TRANSFORMATION_PREPROCESSOR_OBJ = 'preprocessor.pkl'
DATA_TRANSFORMATION_FEATURE_ENGINEERING_OBJ ='feature_eng.pkl' 

DATA_TRANSFORMATION_TRANSFORMER_DIR = 'transformation'
DATA_TRANSFORMATION_TRANSFORMER_TRAIN_DATASET = 'train.csv'
DATA_TRANSFORMATION_TRANSFORMER_TEST_DATASET = 'test.csv'

MODEL_TRAINER_DIR = 'Model_Trainer'
MODEL_TRAINER_OBJ = 'model.pkl'