import numpy as np
import sys

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging

class SequenceCreation:
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length

    def create_sequence(self, x,y):
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

            ## convert lists to numpy arrays
            x_sequence = np.array(x_sequence)
            y_sequence = np.array(y_sequence)

            logging.info('Sequence creation completed')

            logging.info(
                f"X sequence shape: {x_sequence.shape}, "
                f"Y sequence shape: {y_sequence.shape}"
            )

            return(
                x_sequence,
                y_sequence
            )

        except Exception as e:
            logging.exception(
                "Error ocurred during sequence creation"
            )

            raise MARKETPULSEEXCEPTION(e, sys)
