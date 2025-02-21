import sys
import pandas as pd

from src.customer_churn_prediction.config import DataTransformationConfig
from src.customer_churn_prediction.config import ModelTrainerConfig
from src.customer_churn_prediction.utils import load_object

from src.customer_churn_prediction.exception import MyException
from src.customer_churn_prediction.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    # Function to give the prediction and it's probability
    def predict(self, features):
        try:
            preprocessor_path_object = DataTransformationConfig()
            preprocessor_path = preprocessor_path_object.preprocessor_file_path

            model_path_object = ModelTrainerConfig()
            model_path = model_path_object.trained_model_file_path

            model = load_object(file_path='artifacts/model.pkl')
            preprocessor = load_object(file_path='artifacts/preprocessor.pkl')

            scaled_data = preprocessor.transform(features)
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)[0]

            return prediction, probability
        
        except Exception as e:
            raise MyException(e, sys)


class CustomData:
    def __init__(self, 
                 credit_score: int,
                 geography: str,
                 gender: str,
                 age: int,
                 tenure: int,
                 balance: float,
                 num_of_products: int,
                 has_credit_card: int,
                 is_active_member: int,
                 estimated_salary: float):
        self.credit_score = credit_score
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.num_of_products = num_of_products
        self.has_credit_card = has_credit_card
        self.is_active_member = is_active_member
        self.estimated_salary = estimated_salary

    # Function to convert the data to dataframe
    def data_to_dataframe(self):
        try:
            custom_data_input_dict = {'CreditScore':[self.credit_score],
                                      'Geography':[self.geography],
                                      'Gender':[self.gender],
                                      'Age':[self.age],
                                      'Tenure':[self.tenure],
                                      'Balance':[self.balance],
                                      'NumOfProducts':[self.num_of_products],
                                      'HasCrCard':[self.has_credit_card],
                                      'IsActiveMember':[self.is_active_member],
                                      'EstimatedSalary':[self.estimated_salary]
                                      }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise MyException(e, sys)