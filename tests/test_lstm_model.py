from MarketPulse_AI_project.components.data_preprocessing import DataPreprocessing
from MarketPulse_AI_project.components.sequence_creation import SequenceCreation
from MarketPulse_AI_project.models.lstm_model import LSTMModel


if __name__ == "__main__":

    train_path = (
        "project_data/train/train.csv"
    )

    validation_path = (
        "project_data/validation/validation.csv"
    )

    test_path = (
        "project_data/test/test.csv"
    )

    # --------------------------------------------------
    # 1. Data preprocessing
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 2. Sequence creation
    # --------------------------------------------------

    sequence_creator = SequenceCreation(
        sequence_length=30
    )

    X_train_seq, y_train_seq = (
        sequence_creator.create_sequence(
            X_train_scaled,
            y_train
        )
    )

    X_validation_seq, y_validation_seq = (
        sequence_creator.create_evaluation_sequences(
            X_train_scaled,
            X_validation_scaled,
            y_validation
        )
    )

    X_test_seq, y_test_seq = (
        sequence_creator.create_evaluation_sequences(
            X_validation_scaled,
            X_test_scaled,
            y_test
        )
    )

    # --------------------------------------------------
    # 3. Create LSTM model
    # --------------------------------------------------

    lstm_model = LSTMModel()

    # --------------------------------------------------
    # 4. Train
    # --------------------------------------------------

    history = lstm_model.train_model(
        X_train_seq,
        y_train_seq,
        X_validation_seq,
        y_validation_seq
    )

    # --------------------------------------------------
    # 5. Validation prediction
    # --------------------------------------------------

    validation_predictions, validation_probabilities = (
        lstm_model.predict(
            X_validation_seq
        )
    )

    # --------------------------------------------------
    # 6. Validation evaluation
    # --------------------------------------------------

    validation_metrics = lstm_model.evaluate(
        y_validation_seq,
        validation_predictions,
        validation_probabilities
    )

    print("\nValidation Metrics:")
    print(validation_metrics)

    # --------------------------------------------------
    # 7. Test prediction
    # --------------------------------------------------

    test_predictions, test_probabilities = (
        lstm_model.predict(
            X_test_seq
        )
    )

    # --------------------------------------------------
    # 8. Test evaluation
    # --------------------------------------------------

    test_metrics = lstm_model.evaluate(
        y_test_seq,
        test_predictions,
        test_probabilities
    )

    print("\nTest Metrics:")
    print(test_metrics)