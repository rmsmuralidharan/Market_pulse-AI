import sys

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging


class XGBoostModel:
    def __init__(self):

        self.model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss"
        )


    def train_model(self, x_train, y_train):
        try:
            logging.info("starting XGBoost training")

            logging.info(f"Training shape: {x_train.shape}")

            self.model.fit(
                x_train,
                y_train
            )

            logging.info("XGBoost model training completed")
        except Exception as e:
            logging.exception(
                "Error occurred during XGBoost model training"
            )

            raise MARKETPULSEEXCEPTION(e, sys)

    def predict(self, x):
        try:
            logging.info(
                f"Generating XGBoost prediction for shape: {x.shape}"
            )

            prediction = self.model.predict(x)

            probabilities = self.model.predict_proba(x)[:,1]

            return prediction, probabilities
        
        except Exception as e:
            logging.exception(
                "Error occurred during XGBoost prediction"
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
                "Evaluating XGBoost model"
            )

            accuracy = accuracy_score(
                y_true=y_true,
                y_pred=predictions
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
                f"Accuracy: {accuracy:.4f}\n"
                f"Classification Report:\n{class_report}\n"
                f"ROC-AUC: {roc_auc:.4f}\n"
                f"Confusion Matrix:\n{matrix}"
            )

            return {
                "accuracy": accuracy,
                "classification_report": class_report,
                "roc_auc": roc_auc,
                "confusion_matrix": matrix
            }
        
        except Exception as e:
            logging.exception(
                "Error occurred during XGBoost evaluation"
            )

            raise MARKETPULSEEXCEPTION(e, sys)
            
        
