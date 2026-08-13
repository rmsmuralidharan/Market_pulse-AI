from MarketPulse_AI_project.components.data_preprocessing import DataPreprocessing
from MarketPulse_AI_project.components.sequence_creation import SequenceCreation
from MarketPulse_AI_project.models.baseline_model import BaselineModel

if __name__ == "__main__":
    # --------------------------------------------------
    # 1. Data paths
    # --------------------------------------------------

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
    # 2. Data preprocessing
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
    # 3. Create sequences
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
    # 4. Extract the last timestep
    # --------------------------------------------------

    x_train_baseline = X_train_seq[:,-1,:]
    x_validation_baseline = X_validation_seq[:,-1,:]
    x_test_baseline = X_test_seq[:,-1,:]

    # --------------------------------------------------
    # 5. Create baseline model
    # --------------------------------------------------

    baseline_model = BaselineModel()

    # --------------------------------------------------
    # 6. Train
    # --------------------------------------------------

    baseline_model.train_model(
        x_train_baseline,
        y_train_seq
    )

    # --------------------------------------------------
    # 7. Validation prediction
    # --------------------------------------------------

    validation_predictions, validation_probabilities = (
        baseline_model.predict(
            x_validation_baseline
        )
    )

    # --------------------------------------------------
    # 8. Validation evaluation
    # --------------------------------------------------

    validation_metrics = baseline_model.evaluate(
        y_validation_seq,
        validation_predictions,
        validation_probabilities
    )

    print(']nValidation Metrics:')
    print(validation_metrics)  

    # --------------------------------------------------
    # 9. Test prediction
    # --------------------------------------------------

    test_predictions, test_probabilities = (
        baseline_model.predict(
            x_test_baseline
        )
    )

    # --------------------------------------------------
    # 10. Test evaluation
    # --------------------------------------------------

    test_metrics = baseline_model.evaluate(
        y_test_seq,
        test_predictions,
        test_probabilities
    )

    print('\n Test Metrics:')
    print(test_metrics)