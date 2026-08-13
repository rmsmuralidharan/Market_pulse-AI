import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging

class BaselineModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    def train_model(self, x_train, y_train):
        try:
            logging.info("Starting baseline model training")

            logging.info(f"Training input shape: {x_train.shape}")

            self.model.fit(
                x_train,
                y_train
            )

            logging.info("Baseline model Training Completed")
        except Exception as e:
            logging.exception(
                "Error ocurred during baseline model training"
            )

            raise MARKETPULSEEXCEPTION(e,sys)

    def predict(
            self, x
    ):
        try:
            logging.info(
                f"Generating prediction for shape: {x.shape}"
            )

            prediction = self.model.predict(x)

            probabilities = self.model.predict_proba(x)[:,1]

            return prediction, probabilities
        
        except Exception as e:
            logging.exception(
                "Error ocurred during prediction"
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
                "Evaluating the baseline model"
            )

            accuracy = accuracy_score(
                y_true,
                predictions
            )

            class_report = classification_report(
                y_true=y_true,
                y_pred=predictions
            )

            roc_auc = roc_auc_score(
                y_true,
                probabilities
            )

            matrix = confusion_matrix(
                y_true=y_true,
                y_pred=predictions
            )

            logging.info(
                f"Accuracy: {accuracy}, "
                f"Classification Report: {class_report:}, "
                f"ROC-AUC: \n{roc_auc:.4f}, "
                f"Confusion matrix: \n{matrix:}"
            )
            return{
                "accuracy": accuracy,
                "classification_report": class_report,
                "roc_auc": roc_auc,
                "confusion_matrix": matrix
            }
        
        except Exception as e:
            logging.exception(
                "Error ocurred during model evalution"
            )

            raise MARKETPULSEEXCEPTION(e, sys)