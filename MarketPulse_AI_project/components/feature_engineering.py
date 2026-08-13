import os
import sys
import pandas as pd
import numpy as np

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging

class FeatureEngineering:
    def __init__(self, transformed_data_path: str):
        self.transformed_data_path = transformed_data_path

        self.feature_data_dir = "project_data/processed/nifty50"

        self.feature_data_path = os.path.join(
            self.feature_data_dir,
            "nifty50_features.csv"
        )

    def initiate_feature_engineering(self):
        try:
            logging.info("Starting feature engineering")

            df = pd.read_csv(
                self.transformed_data_path
            )

            logging.info(f"Loaded {len(df)} records for feature engineering")

            df['Date'] = pd.to_datetime(
                df['Date']
            )

            df['Daily_Return'] = (
                df['Close'].pct_change()
            )

            df['Log_Return'] = np.log(
                df['Close'] / df['Close'].shift(1)
            )

            df["High_Low_Range"] = (
                (df['High'] - df['Low'])
                / df['Close']
            )

            df['Open_Close_Change'] = (
                (df['Close'] - df['Open'])
                /df['Open']
            )

            ## simple moving average
            df['SMA_5'] = (
                df['Close']
                .rolling(window=5)
                .mean()
            )

            df['SMA_20'] = (
                df['Close']
                .rolling(window=20)
                .mean()
            )

            df['SMA_50'] = (
                df['Close']
                .rolling(window=50)
                .mean()
            )

            ## exponential moving average
            df['EMA_20'] = (
                df['Close']
                .ewm(
                    span=20,
                    adjust=False
                )
                .mean()
            )

            df['EMA_50'] = (
                df['Close']
                .ewm(
                    span=50,
                    adjust=False
                ).mean()
            )

            ## price / trend ratios
            df['Price_SMA20_Ratio'] = (
                df['Close'] / df['SMA_20']
            )

            df['Price_SMA50_Ratio'] = (
                df['Close'] / df['SMA_50']
            )

            df['SMA20_SMA50_Ratio'] = (
                df['SMA_20'] / df['SMA_50']
            )

            ## momentum
            df['Momentum_10'] = (
                df['Close']
                / df['Close'].shift(10)
            ) - 1

            ## RSI - relative strength index
            price_change = df['Close'].diff() 

            gain = price_change.clip(
                lower=0
            )

            loss = - price_change.clip(
                upper=0
            )

            average_gain = (
                gain
                .rolling(window=14)
                .mean()
            )

            average_loss = (
                loss
                .rolling(window=14)
                .mean()
            )

            relative_strength = (
                average_gain / average_loss
            )

            df['RSI_14'] = (
                100 - (100 / (1 + relative_strength))
            )

            ## MACD = difference between a fast (12-day) and slow (26-day) EMA
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()

            df['MACD'] = ema_12 - ema_26

            df['MACD_Signal'] = (
                df['MACD']
                .ewm(
                    span=9,
                    adjust=False
                ).mean()
            )

            df['MACD_Histogram'] = (
                df['MACD'] - df['MACD_Signal']
            )

            ## volatality

            df['Volatility_10'] = (
                df['Log_Return']
                .rolling(window=10)
                .std()
            )

            df['Volatility_20'] = (
                df['Log_Return']
                .rolling(window=20)
                .std()
            )

            ## ATR - average true range

            previous_close = df['Close'].shift(1)

            true_range_1 = df['High'] - df['Low']
            true_range_2 = (df['High'] - previous_close.abs())
            true_range_3 = (df['Low'] - previous_close.abs())

            true_range = pd.concat(
                [true_range_1, true_range_2, true_range_3],
                axis=1
            ).max(axis=1)

            df['ATR_14'] = (
                true_range
                .rolling(window=14)
                .mean()
            )

            ## volume features
            df['Volume_Change'] = (
                df['Volume']
                .pct_change()
            )
            
            df['Volume_SMA_20'] = (
                df['Volume']
                .rolling(window=20)
                .mean()
            )


            df['Relative_Volume'] = (
                df['Volume'] / df['Volume_SMA_20']
            )

            ## lag features
            df['Return_Lag_1'] = (
                df['Daily_Return'].shift(1)
            )

            df['Return_Lag_2'] = (
                df['Daily_Return'].shift(2)
            )

            df['Return_Lag_3'] = (
                df['Daily_Return'].shift(3)
            )

            df['Return_Lag_5'] = (
                df['Daily_Return'].shift(5)
            )

            df['Volume_Lag_1'] = (
                df['Volume'].shift(1)
            )

            ## select final features

            selected_columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",

                "Daily_Return",
                "Log_Return",
                "High_Low_Range",
                "Open_Close_Change",

                "SMA_5",
                "SMA_20",
                "SMA_50",

                "EMA_20",
                "EMA_50",

                "Price_SMA20_Ratio",
                "Price_SMA50_Ratio",
                "SMA20_SMA50_Ratio",

                "Momentum_10",

                "RSI_14",

                "MACD",
                "MACD_Signal",
                "MACD_Histogram",

                "Volatility_10",
                "Volatility_20",

                "ATR_14",

                "Volume_Change",
                "Volume_SMA_20",
                "Relative_Volume",

                "Return_Lag_1",
                "Return_Lag_2",
                "Return_Lag_3",
                "Return_Lag_5",
                "Volume_Lag_1",

                "Target"
            ]

            df = df[selected_columns]


            ## remove rows where historical features are unavailable
            initial_records = len(df)

            df = (
                df.dropna()
                .reset_index(drop=True)
            )

            removed_records = (
                initial_records - len(df)
            )

            logging.info(
                f"Removed {removed_records}"
                f"due to feature generation"
            )

            ## final leakage check

            if df.isnull().sum().sum() > 0:

                raise ValueError(
                    "NaN values reamin after feature engineering"
                )

            ## save feature dataset

            os.makedirs(
                self.feature_data_dir,
                exist_ok=True
            )

            df.to_csv(
                self.feature_data_path,
                index=False
            )

            logging.info(
                "Feature engineering completed successfully"
            )

            logging.info(
                f"feature dataset saved at: "
                f"{self.feature_data_path}"
            )

            logging.info(
                f"Final feature dataset shape: {df.shape}"
            )

            return self.feature_data_path

        except Exception as e:

            logging.exception(
                "Error ocurred during festure engineering"
            )

            raise MARKETPULSEEXCEPTION(e, sys)