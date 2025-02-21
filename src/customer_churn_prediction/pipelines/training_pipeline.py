import sys
import os
from src.customer_churn_prediction.logger import logging
from src.customer_churn_prediction.exception import MyException
from src.customer_churn_prediction.components.data_ingestion import DataIngestion
from src.customer_churn_prediction.components.data_transformation import DataTransformation
from src.customer_churn_prediction.components.model_trainer import ModelTrainer
from src.customer_churn_prediction.config import ModelTrainerConfig

model_trainer = ModelTrainerConfig()
MODEL_PATH = model_trainer.trained_model_file_path


class TrainingPipeline:
    def __init__(self):
        pass
    
    # Function to start the trainig pipeline
    def initiate_training_pipeline(self):
        logging.info("The process has started....")
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Training the model...")
            print('\nIt will take few minutes or more depending on your machine....')
            try:
                data_ingestion = DataIngestion()
                train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

                data_transformation = DataTransformation()
                train_array, test_array, file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

                model_trainer = ModelTrainer()
                model_trainer.initiate_model_trainer(train_array=train_array, test_array=test_array)

                print('Model training is done.')

            except Exception as e:
                logging.info('Raised my exception')
                raise MyException(e, sys)
        else:
            print("Model already trained. Skipping training.")

        