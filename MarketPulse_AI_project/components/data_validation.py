import os
import sys

import pandas as pd
from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging
from MarketPulse_AI_project.entity.artifact_entity import DataValidationArtifact


class DataValidation:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            df = pd.read_csv(self.raw_data_path)

            logging.info(
                f"loaded raw data with {len(df)} records"
            )

            required_columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ] 

            if missing_columns:
                validation_status = False

                message = (
                    f"Missing required columns: {missing_columns}"
                )

                logging.error(message)

                return DataValidationArtifact(
                    validation_status=validation_status,
                    message=message
                )

            ## missing values
            missing_values = df[required_columns].isnull().sum()

            if missing_values.any():
                validation_status = False
                message = (
                    f"Missing values detected: "
                    f"{missing_values[missing_values>0].to_dict()}"
                )

                logging.error(message)

                return DataValidationArtifact(
                    validation_status=validation_status,
                    message=message
                )

            ## duplicate dates check
            duplicate_dates = df['Date'].duplicated().sum()

            if duplicate_dates > 0:
                validation_status = False

                message = (
                    f"Found {duplicate_dates} duplicate dates"
                )

                logging.error(message)

                return DataValidationArtifact(
                    validation_status=validation_status,
                    message=message
                )

            ## date ordering check
            df['Date'] = pd.to_datetime(df['Date'])

            if not df["Date"].is_monotonic_increasing:
                validation_status = False

                message = (
                    "Dates are not in ascending order"
                )

                logging.error(message)

                return DataValidationArtifact(
                    validation_status=validation_status,
                    message=message
                )

            ## invalid prices
            price_columns = [
                        "Open",
                        "High",
                        "Low",
                        "Close"
                    ]
            invalid_prices = (
                df[price_columns] <= 0
            ).any().any()

            if invalid_prices:
                validation_status = False

                message = (
                    "Invalid prices vales detected"
                )

                logging.error(message)

                return DataValidationArtifact(
                    validation_status=validation_status,
                    message=message
                )

            ##  open high low close Consistency
            invalid_ohlc = (
                (df["High"] < df["Open"]) |
                (df["High"] < df["Close"]) |
                (df['Low'] > df["Open"]) |
                (df["Low"] > df["Close"])
            )

            if invalid_ohlc.any():
                validation_status = False

                message = (
                    "Invalid OHLC relationships detected"
                )

                logging.error(message)

                return DataValidationArtifact(
                    validation_status=validation_status,
                    message=message
                )

            ## final validation
            validation_status = True
            message = "Data Validation successful"

            logging.info(message)

            return DataValidationArtifact(
                validation_status= validation_status,
                message=message
            )



        
        

        except Exception as e:
            logging.exception(
                "Error occurred during data validation"
            )

            raise MARKETPULSEEXCEPTION(e, sys)
