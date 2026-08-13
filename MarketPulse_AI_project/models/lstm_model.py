import sys

import numpy as np
from  tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging

class LSTMModel:
    def __init__(self):
        self.model = Sequential([
            LSTM(
                64,
                input_shape = (30,33)
            ),

            Dropout(0.2),

            Dense(
                32,
                activation="relu"
            ),

            Dense(
                1,
                activation="sigmoid"
            )
        ])

        self.model.compile(
            optimizer = Adam(
                learning_rate = 0.001
            ),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

    def train_model(
            self, 
            x_train,
            y_train,
            x_validation,
            y_validation
    ):
        try:
        
            logging.info(
                "Starting LSTM model training"
            )

            logging.info(
                f"Training input shape: {x_train.shape}"
            )

            logging.info(
                f"Validation input shape: "
                f"{x_validation.shape}"
            )

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = self.model.fit(
                x_train,
                y_train,
                validation_data = (
                    x_validation,
                    y_validation
                ),
                epochs = 30,
                batch_size = 32,
                callbacks = [early_stopping],
                verbose =1
            )

            logging.info(
                "LSTM model training completed"
            )

            return history

        except Exception as e:

            logging.exception(
                "Error occurred during LSTM model training"
            )

            raise MARKETPULSEEXCEPTION(e, sys)

    def predict(self, x):
        try:
            logging.info(
                f"Generating LSTM prediction "
                f"for shape: {x.shape}"
            )

            probabilities = (
                self.model.predict(
                    x,
                    verbose = 0
                ).ravel()
            )

            prediction = (
                probabilities >= 0.5
            ).astype(int)

            return prediction, probabilities
        
        except Exception as e:
            logging.exception(
                "Error ocurred during LSTM prediction"
            )

            raise MARKETPULSEEXCEPTION(e, sys)

    def evaluate(
            self,
            y_true,
            predictions,
            probabilities
    ):
        try:
            logging.info(
                "Evaluating LSTM model"
            )

            accuracy = accuracy_score(
                y_true,
                predictions
            )

            class_report = classification_report(
                y_true,
                predictions
            )

            roc_auc = roc_auc_score(
                y_true,
                probabilities
            )

            matrix = confusion_matrix(
                y_true,
                predictions
            )

            logging.info(
                f"Accuracy: {accuracy:.4f}\n"
                f"Classification Report:\n"
                f"{class_report}\n"
                f"ROC-AUC: {roc_auc:.4f}\n"
                f"Confusion Matrix:\n"
                f"{matrix}"
            )

            return {
                "accuracy": accuracy,
                "classification_report": class_report,
                "roc_auc": roc_auc,
                "confusion_matrix": matrix
            }

        except Exception as e:

            logging.exception(
                "Error occurred during LSTM evaluation"
            )

            raise MARKETPULSEEXCEPTION(e, sys)
        