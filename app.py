import sys
from src.customer_churn_prediction.logger import logging
from src.customer_churn_prediction.exception import MyException

if __name__ == '__main__':
    logging.info("The process has started....")

    try:
        a=1/0
    except Exception as e:
        logging.info('Raised my exception')
        raise MyException(e, sys)