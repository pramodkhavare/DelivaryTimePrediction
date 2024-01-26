# from src.constant import * 
# from src.config.configuration import *

# from src.logger import logging 

# from src.pipeline.prediction_pipeline import CustomData ,PredictionPipeline
# from src.pipeline.prediction_pipeline import PredictionPipeline
# from src.pipeline.training_pipeline import Train
# import os ,sys 
# from flask import Flask,render_template ,request 
# from werkzeug.utils import secure_filename


# app = Flask(__name__ ,template_folder = 'templates')

# @app.route('/')
# def home_page():
#     return render_template('index.html')




# @app.route('/predict' ,methods = ['GET' , 'POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('form.html')
    
    
#     else:
#         data = CustomData(
#             Delivery_person_Age = float(request.form.get('Delivery_person_Age')),
#             Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
#             Vehicle_condition = float(request.form.get('Vehicle_condition')),
#             multiple_deliveries = float(request.form.get('multiple_deliveries')),
#             Distance = float(request.form.get('Distance')),
#             Type_of_order =request.form.get('Type_of_order'),
#             Type_of_vehicle = request.form.get('Type_of_vehicle'),
#             Festival = request.form.get('Festival'),
#             City = request.form.get('City'),
#             Weather_conditions = request.form.get('Weather_conditions'),
#             Road_traffic_density = request.form.get('Road_traffic_density')
#         )
        
#         final_new_data = data.get_data_as_dataframe()
#         predict_pipeline = PredictionPipeline()
#         pred = predict_pipeline.predict(final_new_data)
        
#         result = int(pred[0])
        
#         return render_template("form.html",final_result = "Your Delivery Time IS. {}".format(result))
        
# if  __name__ == '__main__':
#     app.run(host= '0.0.0.0' ,debug=True ,port='8000')


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

        
        
        
        
        
if  __name__ == '__main__':
    app.run(host= '0.0.0.0' ,debug=True ,port='8000')