import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.Data_transformation import DataTransformation
from src.components.Data_transformation import DataTransformationConfig

from src.components.Model_Trainer import ModelTrainerConfig
from src.components.Model_Trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    test_data_path: str = os.path.join("artifacts", "test_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into the data ingestion component")
        try:
            df = pd.read_csv("../../Notebook/Data/stud.csv")
            logging.info("read the raw dataset as a dataframe")

            os.makedirs("artifacts", exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info("train_test_split initiated")
            test_set, train_set = train_test_split(df, test_size=0.2, random_state=42)

            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)

            logging.info("ingestion of the data is completed")

            return(
                self.data_ingestion_config.test_data_path,
                self.data_ingestion_config.train_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _=data_transformation.initiate_data_transformation(train_data,test_data)

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    print(train_arr)