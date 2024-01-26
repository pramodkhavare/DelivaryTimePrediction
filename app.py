from src.constant import * 
from src.config.configuration import *

from src.logger import logging 

from src.pipeline.prediction_pipeline import CustomData ,PredictionPipeline
from src.pipeline.training_pipeline import Train

import os ,sys 
from flask import Flask,render_template ,request 
from werkzeug.utils import secure_filename


processor_file_path = PREPROCESSOR_OBJ_FILE_PATH
model_file_path = MODEL_TRAINER_FILE_PATH

UPLOAD_FOLDER = 'batch_prediction/UPLOADED_CSV_FILE'  #WE WILL STORE UPLOADED FILE DATA IN THESE FOLDER

app = Flask(__name__ ,template_folder = 'templates')

ALLOWED_EXTENSION = {'csv'}





@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict' ,methods = ['GET' , 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    
    else:
        data = CustomData(
            Delivery_person_Age = float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Vehicle_condition = float(request.form.get('Vehicle_condition')),
            multiple_deliveries = float(request.form.get('multiple_deliveries')),
            Distance = float(request.form.get('Distance')),
            Type_of_order =request.form.get('Type_of_order'),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            Festival = request.form.get('Festival'),
            City = request.form.get('City'),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density')
        )
        
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        result = int(pred[0])
        
        return render_template("form.html",final_result = "Your Delivery Time IS. {}".format(result))


@app.route('/batch' ,methods = ['GET' ,'POST'])
def perform_batch_prediction():
    if request.method == "GET" :
        return render_template('batch.html')
    
    else:
        file = request.files['csv_file']
        
        #Directoey path 
        directory_path = UPLOAD_FOLDER
        #Create Directory 
        os.makedirs(directory_path ,exist_ok=True)
        
        #check file have valid extension
        if file and  '.' in file.filename and file.filename.rsplit('.' ,1)[1].lower() in ALLOWED_EXTENSION:
            #delete all files in file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER ,filename)
                if os.path.isfile(file_path):
                    os.remove(file_path) 
                    
            #Save the new file to the upload directory
            filename =secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER ,filename)
            file.save(file_path)
            print(file_path)
            logging.info('CSV file rcieved and uploaded')
         
            
            output = "Batch Prediction Is Done"
            return render_template('batch.html' , prediction_result = output ,prediction_type = 'batch')
        else:
            return render_template('batch.html' ,prediction_type= 'batch' ,error = 'Invalid file type')


@app.route('/train' ,methods = ['GET' , 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html') 
    
    else:
        try:
            pipeline = Train()
            pipeline.main()
        except Exception as e:
            logging.error(f"{e}")
            error_message =str(e)
            return render_template('index.html' ,error =error_message)     
        
        
        
        
        
        
if  __name__ == '__main__':
    app.run(host= '0.0.0.0' ,debug=True ,port='8000')