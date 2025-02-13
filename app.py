import sys
from src.customer_churn_prediction.logger import logging
from src.customer_churn_prediction.exception import MyException
from src.customer_churn_prediction.components.data_ingestion import DataIngestion
from src.customer_churn_prediction.components.data_transformation import DataTransformation

if __name__ == '__main__':
    logging.info("The process has started....")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    except Exception as e:
        logging.info('Raised my exception')
        raise MyException(e, sys)