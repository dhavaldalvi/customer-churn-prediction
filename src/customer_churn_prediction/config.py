import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train_data.csv')
    test_data_path:str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')