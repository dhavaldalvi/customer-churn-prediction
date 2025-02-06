import sys
from src.customer_churn_prediction.logger import logging
from src.customer_churn_prediction.exception import MyException
from src.customer_churn_prediction.components.data_ingestion import DataIngestion
from src.customer_churn_prediction.config import DataIngestionConfig

if __name__ == '__main__':
    logging.info("The process has started....")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info('Raised my exception')
        raise MyException(e, sys)