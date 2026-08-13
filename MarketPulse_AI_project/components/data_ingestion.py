import os
import yfinance as yf
from MarketPulse_AI_project.logging.logger import logging
from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
import sys

class DataIngestion:
    def __init__(self):
        self.symbol = "^NSEI"
        self.raw_data_dir = "project_data/raw/nifty50"


    def initiate_data_ingestion(self):
        try:
            logging.info("Starting Nifty 50 data ingestion")

            os.makedirs(self.raw_data_dir, exist_ok=True)

            data = yf.download(
                self.symbol,
                period="5y",
                interval='1d',
                auto_adjust=False,
                multi_level_index=False
            )

            if data.empty:
                raise Exception("No data received from yfinance")

            data.reset_index(inplace=True)

            raw_file_path = os.path.join(
                self.raw_data_dir,
                "nifty50_raw.csv"
            )

            data.to_csv(raw_file_path)

            logging.info(
                f"NIFTY 50 data downloaded succcessfully. "
                f"Records: {len(data)}"
            )

            logging.info(
                f"Raw data saved at: {raw_file_path}"
            )

            return raw_file_path
        
        except Exception as e:
            logging.exception(
                "Error occurred during NIFTY 50 data ingestion" 
            )

            raise MARKETPULSEEXCEPTION(e, sys)