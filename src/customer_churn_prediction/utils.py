from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 
import pickle
import dill
import os
from src.customer_churn_prediction.exception import MyException
import sys
    

# Function to remove outliers for numerical columns
def remove_numerical_outliers(df):
    # Apply IQR for numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the condition for outliers
    condition = ~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # Return the Dataframe
    return df[condition]

# Function to remove outliers for categorical columns
def remove_categorical_outliers(df, threshold=2):
    # Apply frequency-based filtering for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Get value counts and filter categories with less than the threshold frequency
        value_counts = df[col].value_counts()
        rare_values = value_counts[value_counts < threshold].index
        
        # Remove rows where the categorical value is rare
        df = df[~df[col].isin(rare_values)]
    
    # Return the Dataframe
    return df

def save_object(file_path, obj):
    '''
    Function to save pickle object file
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise MyException(e, sys)
    

def best_model_select(df, accuracy_low_limit):
    '''
    Function to return best model from the Dataframe of models
    '''
    cols = df.columns
    if (df[df.columns[0]].max() and df[df.columns[1]].max()) < accuracy_low_limit:
        print('No best model')
        return 
    elif df[df.columns[0]].max() > df[df.columns[1]].max():
        x = df[df[df.columns[1]]==df[df.columns[1]].max()].index[0]
    else:
        x = df[df[df.columns[0]]==df[df.columns[0]].max()].index[0]
    return x

def load_object(file_path):
    '''
    Function to load pickle file
    '''
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise MyException(e, sys)
    

