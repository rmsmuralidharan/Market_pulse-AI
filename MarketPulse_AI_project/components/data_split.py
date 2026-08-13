import os
import pandas as pd
import sys

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging

class DataSplit:
    def __init__(self):
        pass

    def initiate_data_split(self, input_file_path):
        try:
            logging.info("Starting data splitting")

            df = pd.read_csv(input_file_path)

            logging.info(f"Loaded data from: {input_file_path}")
            logging.info(f"Dataset shape: {df.shape}")

            ## making sure the date ordered chronologically
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values("Date").reset_index(drop=True)

            ## chronological split
            train_size = int(len(df) * 0.70)
            val_size = int(len(df) * 0.15)

            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size: train_size + val_size]
            test_df = df.iloc[train_size + val_size:]

            logging.info(
                f"Train shape: {train_df.shape}, "
                f"Validation shaape: {val_df.shape}, "
                f"Test Shape: {test_df.shape}"
            )

            ## output directories
            train_path = os.path.join(
                "project_data", "train", "train.csv"
            )

            val_path = os.path.join(
                "project_data", "validation", "validation.csv"
            )

            test_path = os.path.join(
                "project_data", "test", "test.csv"
            )

            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            os.makedirs(os.path.dirname(val_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_path), exist_ok=True)

            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info(
                "Train/validation/test split completed successfully"
            )

            return train_path, val_path, test_path
        
        except Exception as e:
            logging.error(
                "Error occured during splitting"
            )

            raise MARKETPULSEEXCEPTION(e, sys)