import numpy as np
import pandas as pd
import sys

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

from src.customer_churn_prediction.exception import MyException
from src.customer_churn_prediction.logger import logging

from src.customer_churn_prediction.config import ModelTrainerConfig
from src.customer_churn_prediction.utils import save_object, best_model_select

# Class to initiate model training process to select best model
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # Function to initiate the model training to select best model
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing dataset")
            # Splitting training and testing dataset
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Dictionary of models to use it in GridSearchCV
            models = {
                'LogisticRegression': LogisticRegression(),
                'SVM': SVC(),
                'DecisionTree': DecisionTreeClassifier(),
                'RandomForest': RandomForestClassifier(),
                'GradientBoosting': GradientBoostingClassifier(),
                'XGBClassifier': XGBClassifier()
                }

            # Sets of hyperparameters for each models
            param_grids = {
                'LogisticRegression': {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear', 'saga'],
                    'penalty': ['l2'],
                    'class_weight':['balanced'],
                    'max_iter':[500]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'class_weight':['balanced']
                },
                'DecisionTree': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'class_weight':['balanced']
                },
                'RandomForest': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                    'criterion': ['gini', 'entropy'],
                    'class_weight':['balanced']
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'XGBClassifier':{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'gamma': [0, 0.1, 0.2]
                }
                }

            # Running GridSearchCV on each model
            best_models = {}
            for model_name, model in models.items():

                print(f"\nRunning GridSearchCV for {model_name}...")

                # Setting GrdiSearchCV
                grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, n_jobs=-1, verbose=1)
                # Running GridSearchCV
                grid_search.fit(X_train, y_train)
                # Saving best hyperparameters for each model
                best_models[model_name] = grid_search.best_estimator_

                print(f"\nEvaluating the model: {model_name} with best hyperparameters: {grid_search.best_params_}")
                print('-'*100)

            # Creating lists to store f1-score for each target class
            f1_class_0 = []
            f1_class_1 = []
            index = []
            # Evaluate the best models
            for model_name, model in best_models.items():
                # Predicting test dataset
                y_pred = model.predict(X_test)
                # Generating report
                report_result = classification_report(y_test, y_pred, output_dict=True)
                # Storing the f1-scores for each target class
                f1_class_0.append(report_result['0.0']['f1-score'])
                f1_class_1.append(report_result['1.0']['f1-score'])
                index.append(model_name)

            # Storing f1-scores as dataframe
            model_f1_score = pd.DataFrame({'f1-score_class_0':f1_class_0, 'f1-score_class_1': f1_class_1}, index=index)
            print('\nDataframe of various models with their f-score for class 0 (No) or No and 1 (Yes).')
            print(model_f1_score)
            logging.info(model_f1_score)

            print(f'\nBest model is {best_model_select(model_f1_score, accuracy_low_limit=0.55)}')
            print('-'*100)
            logging.info(f'Best model is {best_model_select(model_f1_score, accuracy_low_limit=0.55)}')

            # Choosing best model
            best_model = models[best_model_select(model_f1_score, accuracy_low_limit=0.55)]

            # Fitting or running the chosen best model
            best_model = best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise MyException(e, sys)