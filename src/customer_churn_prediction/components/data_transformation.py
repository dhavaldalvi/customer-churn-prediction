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

# Class to initiate data transformation process
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Function to get the preprocessor object
    def get_data_transformation_object(self):
        try:
            logging.info("Features preprocessing......")
            # Separating categorical and numerical columns
            categorical_cols = ['Geography', 'Gender']
            numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

            # Setting preprocessor for scaling categorical and numerical features
            preprocessor = ColumnTransformer(
                transformers=[
                                ('numerical scaler', StandardScaler(), numerical_cols),  # Apply StandardScaler to numerical columns
                                ('categorical encoder', OneHotEncoder(), categorical_cols)      # Apply OneHotEncoder to nominal columns
                            ])
            logging.info("Features preprocessed.")

            return preprocessor
        
        except Exception as e:
            raise MyException(e, sys)
    
    # Function to initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function transfroms the data.
        '''
        try:
            # Reading training data
            train_data = pd.read_csv(train_path)
            # Reading testing data
            test_data = pd.read_csv(test_path)

            logging.info("Reading train and test data...")

            # Getting data transfromation object
            preprocesssor_obj = self.get_data_transformation_object()

            # Choosing target column name
            target_feature = 'Exited'

            # Selecting features features training
            input_train_data = train_data.drop(columns=[target_feature], axis = 1)
            # Selecting target feature for traninig
            target_train_data = train_data[target_feature]

            # Selecting features for testing
            input_test_data = test_data.drop(columns=[target_feature], axis = 1)
            # Selecting target features for testing
            target_test_data = test_data[target_feature]

            logging.info("Applying preprocessing...")

            # Transforming or scaling the features
            input_train_data_preprocessed = preprocesssor_obj.fit_transform(input_train_data)

            # Resampling the dataset using SMOTE since the target feature has imbalanced classes
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(input_train_data_preprocessed, target_train_data)

            # Also transfroming and scaling the testing dataset
            X_test_preprocessed = preprocesssor_obj.transform(input_test_data)

            # Setting the training and testing dataset as an array
            train_array = np.c_[X_train_resampled, y_train_resampled]
            test_array = np.c_[X_test_preprocessed, np.array(target_test_data)]

            logging.info("Saved preprocesing")

            # Saving the preprocessor file
            save_object(file_path=self.data_transformation_config.preprocessor_file_path, obj=preprocesssor_obj)

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_file_path
            )


        except Exception as e:
            raise MyException(e,sys)