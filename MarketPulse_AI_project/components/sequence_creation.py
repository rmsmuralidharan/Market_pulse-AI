import numpy as np
import sys

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging


class SequenceCreation:

    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length

    # -----------------------------------------
    # Training sequences
    # -----------------------------------------
    def create_sequence(self, x, y):

        try:

            logging.info("Starting sequence creation")

            logging.info(f"Input x shape: {x.shape}")
            logging.info(f"Input y shape: {y.shape}")

            x_sequence = []
            y_sequence = []

            for i in range(
                self.sequence_length,
                len(x)
            ):

                x_sequence.append(
                    x[
                        i - self.sequence_length:i
                    ]
                )

                y_sequence.append(
                    y[i]
                )

            x_sequence = np.array(x_sequence)
            y_sequence = np.array(y_sequence)

            logging.info(
                "Sequence creation completed"
            )

            logging.info(
                f"X sequence shape: {x_sequence.shape}, "
                f"Y sequence shape: {y_sequence.shape}"
            )

            return (
                x_sequence,
                y_sequence
            )

        except Exception as e:

            logging.exception(
                "Error occurred during sequence creation"
            )

            raise MARKETPULSEEXCEPTION(e, sys)

    # -----------------------------------------
    # Validation / Test sequences
    # -----------------------------------------
    def create_evaluation_sequences(
        self,
        X_history,
        X_current,
        y_current
    ):

        try:

            logging.info(
                "Starting evaluation sequence creation"
            )

            historical_data = X_history[
                -(self.sequence_length - 1):
            ]

            X_combined = np.concatenate(
                [
                    historical_data,
                    X_current
                ],
                axis=0
            )

            X_sequences = []
            y_sequences = []

            for i in range(
                len(X_current)
            ):

                start = i
                end = i + self.sequence_length

                X_sequences.append(
                    X_combined[start:end]
                )

                y_sequences.append(
                    y_current[i]
                )

            X_sequences = np.array(
                X_sequences
            )

            y_sequences = np.array(
                y_sequences
            )

            logging.info(
                f"Evaluation sequence shape: "
                f"{X_sequences.shape}"
            )

            logging.info(
                f"Evaluation target shape: "
                f"{y_sequences.shape}"
            )

            return (
                X_sequences,
                y_sequences
            )

        except Exception as e:

            logging.exception(
                "Error occurred during evaluation sequence creation"
            )

            raise MARKETPULSEEXCEPTION(e, sys)