from flask import Flask, render_template, request
import numpy
import pandas as pd 

from src.customer_churn_prediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.customer_churn_prediction.pipelines.training_pipeline import TrainingPipeline

# import sys
# from src.customer_churn_prediction.logger import logging
# from src.customer_churn_prediction.exception import MyException
# from src.customer_churn_prediction.components.data_ingestion import DataIngestion
# from src.customer_churn_prediction.components.data_transformation import DataTransformation
# from src.customer_churn_prediction.components.model_trainer import ModelTrainer

app = Flask(__name__)

TrainingPipeline().initiate_training_pipeline()

# Home page
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = CustomData(
        credit_score = request.form.get('CreditScore'),
        geography = request.form.get('Geography'),
        gender = request.form.get('Gender'),
        age = request.form.get('Age'),
        tenure = request.form.get('Tenure'),
        balance = request.form.get('Balance'),
        num_of_products = request.form.get('NumOfProducts'),
        has_credit_card = request.form.get('HasCrCard'),
        is_active_member = request.form.get('IsActiveMember'),
        estimated_salary = request.form.get('EstimatedSalary'),
    )
    prediction_data = data.data_to_dataframe()
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(prediction_data)[0]
    probability = predict_pipeline.predict(prediction_data)[1]
    if prediction[0] == 0:
        result = 'No'
        probability = probability[0]
    else:
        result = 'Yes'
        probability = probability[1]

    return render_template('result.html', result=result, probability = round(probability,2))


if __name__ == '__main__':
   app.run(debug=False)


#if __name__ == '__main__':
#    logging.info("The process has started....")

    # try:
    #     data_ingestion = DataIngestion()
    #     train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    #     data_transformation = DataTransformation()
    #     train_array, test_array, file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    #     model_trainer = ModelTrainer()
    #     model_trainer.initiate_model_trainer(train_array=train_array, test_array=test_array)

    # except Exception as e:
    #     logging.info('Raised my exception')
    #     raise MyException(e, sys)