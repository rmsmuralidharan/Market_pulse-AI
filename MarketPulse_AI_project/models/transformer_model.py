import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import(
    Input,
    Dense,
    Dropout,
    LayerNormalization,MultiHeadAttention,
    GlobalAveragePooling1D
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import(
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from MarketPulse_AI_project.exception.exception import MARKETPULSEEXCEPTION
from MarketPulse_AI_project.logging.logger import logging

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
            self,
            embed_dim=64,
            num_heads =4,
            ff_dim = 128,
            drop_rate = 0.2
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.feed_forward = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])

        self.layer_norm1 = LayerNormalization(
            epsilon=1e-6
        )

        self.layer_norm2 = LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = Dropout(
            drop_rate
        )

        self.dropout2 = Dropout(
            drop_rate
        )

    def call(self, inputs, training = False):
        attention_output = self.attention(
            inputs,
            inputs,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        out1 = self.layer_norm1(
            inputs + attention_output
        )

        ff_output = self.feed_forward(
            out1,
            training =training
        )

        ff_output = self.dropout2(
            ff_output,
            training=training
        )

        return self.layer_norm2(
            out1 + ff_output
        )

class TransformerModel:

    def __init__(self):
        inputs = Input(
            shape = (30, 33)
        )

        # Convert 33 features into 64-dimensional representation
        x = Dense(
            64
        )(inputs)

        # Transformer Encoder
        x = TransformerEncoder(
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            drop_rate=0.2
        )(x)

        ## convert 30 timme steps into one representation
        x = GlobalAveragePooling1D()(x)

        x= Dropout(
            0.2
        )(x)

        x = Dense(
            32, 
            activation='relu'
        )(x)

        x = Dropout(
            0.2
        )(x)

        outputs = Dense(
            1,
            activation="sigmoid"
        )(x)

        self.model = Model(
            inputs=inputs,
            outputs=outputs
        )

        self.model.compile(
            optimizer = Adam(
                learning_rate = 0.001
            ),
             loss = "binary_crossentropy",
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
                'Starting Transformer model training'
            )

            logging.info(
                f"Training input shape: "
                f"{x_train.shape}"
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
                epochs = 50,
                batch_size = 32,
                callbacks = [early_stopping],
                verbose = 1
            
            )

            logging.info(
                "Transformer model training completed"
            )

            return history

        except Exception as e:

            logging.exception(
                "Error occurred during Transformer model training"
            )

            raise MARKETPULSEEXCEPTION(e, sys)

    def predict(self, x):

        try:

            logging.info(
                f"Generating Transformer prediction "
                f"for shape: {x.shape}"
            )

            probabilities = (
                self.model.predict(
                    x,
                    verbose=0
                ).ravel()
            )

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            return predictions, probabilities

        except Exception as e:

            logging.exception(
                "Error occurred during Transformer prediction"
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
                "Evaluating Transformer model"
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
                "Error occurred during Transformer evaluation"
            )

            raise MARKETPULSEEXCEPTION(e, sys)

