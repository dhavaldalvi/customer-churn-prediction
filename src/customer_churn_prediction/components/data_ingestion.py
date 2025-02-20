import os
import sys
from src.customer_churn_prediction.logger import logging
from src.customer_churn_prediction.exception import MyException
from src.customer_churn_prediction.config import DataIngestionConfig
import pandas as pd
from sklearn.model_selection import train_test_split

# Class to initiate data ingestion process
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Function to initiate data ingestion
    def initiate_data_ingestion(self):
        try:
            # Reading data from data folder or if required from any database
            data = pd.read_csv('data/Churn_Modelling.csv')
            data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
            logging.info("Reading from data")
            # Making folder to store data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # Storing data as raw data
            data.to_csv(self.ingestion_config.raw_data_path)
            # Splitting raw data into training and testing data set
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=43)
            # Storing training data set to a folder
            train_set.to_csv(self.ingestion_config.train_data_path)
            # Storing testing data set to a folder
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info("Data ingestion is complete")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise MyException(e, sys)