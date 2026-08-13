import pandas as pd

from MarketPulse_AI_project.components.data_preprocessing import DataPreprocessing
from MarketPulse_AI_project.components.sequence_creation import SequenceCreation

if __name__ =="__main__":

    train_path = "project_data/train/train.csv"
    validation_path = "project_data/validation/validation.csv"
    test_path = "project_data/test/test.csv"

    preprocessing = DataPreprocessing(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path
    )

    (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        y_train,
        y_validation,
        y_test,
        scaler
    ) = preprocessing.initiate_data_preprocessing()

    sequence_creator = SequenceCreation(
        sequence_length=30
    )  

    x_train_seq, y_train_seq = (
        sequence_creator.create_sequence(
            X_train_scaled,
            y_train
        )
    )

    x_validation_seq, y_validation_seq = (
        sequence_creator.create_sequence(
        X_validation_scaled,
        y_validation
        )
    )

    x_test_seq, y_test_seq = (
        sequence_creator.create_sequence(
            X_test_scaled,
            y_test
        )
    )

    print("\nTrain sequence shape:")
    print(x_train_seq.shape)

    print("\nTrain target shape:")
    print(y_train_seq.shape)

    print("\nValidation sequence shape:")
    print(x_validation_seq.shape)

    print("\nTest sequence shape:")
    print(x_test_seq.shape)