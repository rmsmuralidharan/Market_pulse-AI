import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging

class DataPreprocessing:
    def __init__(
            self,
            train_path: str,
            validation_path: str,
            test_path: str
    ):
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path

    def initiate_data_preprocessing(self):
        try:
            logging.info("Starting data preprocessing")

            ## load the dataset
            train_df = pd.read_csv(self.train_path)
            validation_df = pd.read_csv(self.validation_path)
            test_df = pd.read_csv(self.test_path)

            logging.info(
                f"Train shape: {train_df.shape}, "
                f"Validataion shape: {validation_df.shape}, "
                f"Test shape: {test_df.shape} "
            )

            ## seperate features and target
            target_column = 'Target'
            x_train = train_df.drop(
                columns=[target_column, "Date"]
            )

            y_train = train_df[target_column]

            x_validation = validation_df.drop(
                columns=[target_column, 'Date']
            )

            y_validation = validation_df[target_column]

            x_test = test_df.drop(
                columns=[target_column, "Date"]
            )

            y_test = test_df[target_column]

            # create scaler
            scaler = StandardScaler()

            ## fit only on training data
            x_train_scaled = scaler.fit_transform(x_train)

            ### transform validation and test
            x_validation_scaled = scaler.transform(x_validation)

            x_test_scaled = scaler.transform(x_test)

            logging.info(
                "Feature scaling completed successfully"
            )

            return(
                x_train_scaled,
                x_validation_scaled,
                x_test_scaled,
                y_train.to_numpy(),
                y_validation.to_numpy(),
                y_test.to_numpy(),
                scaler
            )
        except Exception as e:
            logging.exception(
                "Error ocurred during data preprocessing"
            )

            raise MARKETPULSEEXCEPTION(e, sys)