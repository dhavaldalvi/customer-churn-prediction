import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.customer_churn_prediction.exception import MyException
from src.customer_churn_prediction.logger import logging

from src.customer_churn_prediction.config import DataTransformationConfig
from src.customer_churn_prediction.utils import save_object, remove_numerical_outliers

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function performs data transfromation
        '''
        try:
            logging.info("Features preprocessing......")
            categorical_cols = ['Geography', 'Gender']
            numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            preprocessor = ColumnTransformer(
                transformers=[
                                ('numerical scaler', StandardScaler(), numerical_cols),  # Apply StandardScaler to numerical columns
                                ('categorical encoder', OneHotEncoder(), categorical_cols)      # Apply OneHotEncoder to nominal columns
                            ])
            logging.info("Features preprocessed.")

            return preprocessor
        
        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Reading train and test data...")

            preprocesssor_obj = self.get_data_transformation_object()

            target_feature = 'Exited'

            input_train_data = train_data.drop(columns=[target_feature], axis = 1)
            target_train_data = train_data[target_feature]

            input_test_data = test_data.drop(columns=[target_feature], axis = 1)
            target_test_data = test_data[target_feature]

            logging.info("Applying preprocessing...")

            input_train_data_preprocessed = preprocesssor_obj.fit_transform(input_train_data)

            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(input_train_data_preprocessed, target_train_data)

            X_test_preprocessed = preprocesssor_obj.transform(input_test_data)

            train_array = np.c_[X_train_resampled, y_train_resampled]
            test_array = np.c_[X_test_preprocessed, np.array(target_test_data)]

            logging.info("Saved preprocesing")

            save_object(file_path=self.data_transformation_config.preprocessor_file_path, obj=preprocesssor_obj)

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_file_path
            )


        except Exception as e:
            raise MyException(e,sys)