from MarketPulse_AI_project.components.data_preprocessing import DataPreprocessing
from MarketPulse_AI_project.components.sequence_creation import SequenceCreation
from MarketPulse_AI_project.models.xgboost_model import XGBoostModel


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
    # Data preprocessing
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
    # Sequence creation
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
        sequence_creator.create_sequence(
            X_validation_scaled,
            y_validation
        )
    )

    X_test_seq, y_test_seq = (
        sequence_creator.create_sequence(
            X_test_scaled,
            y_test
        )
    )

    # --------------------------------------------------
    # Last timestep for tabular XGBoost
    # --------------------------------------------------

    X_train_xgb = X_train_seq[:, -1, :]
    X_validation_xgb = X_validation_seq[:, -1, :]
    X_test_xgb = X_test_seq[:, -1, :]

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    xgb_model = XGBoostModel()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    xgb_model.train_model(
        x_train=X_train_xgb,
        y_train=y_train_seq
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    validation_prediction, validation_probabilities = (
        xgb_model.predict(
            X_validation_xgb
        )
    )

    validation_metrics = xgb_model.evaluate(
        y_validation_seq,
        validation_prediction,
        validation_probabilities
    )

    print("\nValidation Metrics:")
    print(validation_metrics)


   # --------------------------------------------------
    # Test
    # --------------------------------------------------

    test_predictions, test_probabilities = (
        xgb_model.predict(
            X_test_xgb
        )
    )

    test_metrics = xgb_model.evaluate(
        y_test_seq,
        test_predictions,
        test_probabilities
    )

    print("\nTest Metrics:")
    print(test_metrics)