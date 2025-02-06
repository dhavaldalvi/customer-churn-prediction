import os
import sys
from src.customer_churn_prediction.logger import logging
from src.customer_churn_prediction.exception import MyException
from src.customer_churn_prediction.config import DataIngestionConfig
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            data = pd.read_csv('data/Churn_Modelling.csv')
            logging.info("Reading from data")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path)
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=43)
            data.to_csv(self.ingestion_config.train_data_path)
            data.to_csv(self.ingestion_config.test_data_path)
            logging.info("Data ingestion is complete")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise MyException(e, sys)