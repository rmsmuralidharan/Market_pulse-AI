from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging

import sys
import os
import pandas as pd

class DataTransformation:

    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation")

            df = pd.read_csv(self.raw_data_path)
            df['Date'] = pd.to_datetime(df['Date'])

            ## sort Chronologically
            df= df.sort_values("Date").reset_index(drop=True)

            logging.info(
                f"Loaded {len(df)} records for transformation"
            )

            ## handling missing values
            missing_values = df.isnull().sum()

            if missing_values.any():
                message = (
                    f"Missing values detected: "
                    f"{missing_values[missing_values>0].to_dict()}"
                )

                logging.error(message)
                raise ValueError(message)

            logging.info("No missing values detected")

            ## duplicate date handling
            duplicate_dates = df['Date'].duplicated().sum()

            if duplicate_dates > 0:

                message = (
                    f"Found {duplicate_dates} duplicate dates"
                )

                logging.error(message)
                raise ValueError(message)

            logging.info("No duolicates dates detected")

            ## data type convertion check for numerical columns
            numeric_columns = [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]
            for column in numeric_columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="raise"
                )

            logging.info("Numerical columns are validated successfully")


            # Target: next trading day's direction
            df['Target'] = (
                df['Close'].shift(-1) > df['Close']
            ).astype(int)

            ## last row has no future day to compare against
            df= df.iloc[:-1].copy()

            logging.info(
                "Target column created successfully"
            )

            selected_columns = [
                        "Date",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "Target"
                    ]
            df = df[selected_columns]

            ## save transformed data
            processed_data_dir = "project_data/processed/nifty50"

            os.makedirs(
                processed_data_dir,
                exist_ok=True
            )

            processed_file_path = os.path.join(
                processed_data_dir,
                "nifty50_transformed.csv"
            )

            df.to_csv(
                processed_file_path,
                index=False
            )

            logging.info(
                f"Transformed data saved at: {processed_file_path}"
            )

            return processed_file_path

            

            return df
        except Exception as e:

            logging.exception(
                "Error ocurred during data transformation"
            )

            raise MARKETPULSEEXCEPTION(e, sys)