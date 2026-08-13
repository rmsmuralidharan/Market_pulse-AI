from MarketPulse_AI_project.components.data_preprocessing import DataPreprocessing
from MarketPulse_AI_project.components.sequence_creation import SequenceCreation
from MarketPulse_AI_project.models.transformer_model import TransformerModel


if __name__ == "__main__":

    # ==================================================
    # 1. Data paths
    # ==================================================

    train_path = (
        "project_data/train/train.csv"
    )

    validation_path = (
        "project_data/validation/validation.csv"
    )

    test_path = (
        "project_data/test/test.csv"
    )

    # ==================================================
    # 2. Data preprocessing
    # ==================================================

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

    # ==================================================
    # 3. Sequence creation
    # ==================================================

    sequence_creator = SequenceCreation(
        sequence_length=30
    )

    # --------------------------------------------------
    # Training sequences
    # --------------------------------------------------

    X_train_seq, y_train_seq = (
        sequence_creator.create_sequence(
            X_train_scaled,
            y_train
        )
    )

    # --------------------------------------------------
    # Validation sequences
    # Use previous training data as historical context
    # --------------------------------------------------

    X_validation_seq, y_validation_seq = (
        sequence_creator.create_evaluation_sequences(
            X_train_scaled,
            X_validation_scaled,
            y_validation
        )
    )

    # --------------------------------------------------
    # Test sequences
    # Use previous validation data as historical context
    # --------------------------------------------------

    X_test_seq, y_test_seq = (
        sequence_creator.create_evaluation_sequences(
            X_validation_scaled,
            X_test_scaled,
            y_test
        )
    )

    # ==================================================
    # 4. Display sequence shapes
    # ==================================================

    print("\nTrain sequence shape:")
    print(X_train_seq.shape)

    print("\nValidation sequence shape:")
    print(X_validation_seq.shape)

    print("\nTest sequence shape:")
    print(X_test_seq.shape)

    print("\nTrain target shape:")
    print(y_train_seq.shape)

    print("\nValidation target shape:")
    print(y_validation_seq.shape)

    print("\nTest target shape:")
    print(y_test_seq.shape)

    # ==================================================
    # 5. Create Transformer model
    # ==================================================

    transformer_model = TransformerModel()

    # ==================================================
    # 6. Train Transformer
    # ==================================================

    transformer_model.train_model(
        X_train_seq,
        y_train_seq,
        X_validation_seq,
        y_validation_seq
    )

    # ==================================================
    # 7. Validation prediction
    # ==================================================

    (
        validation_predictions,
        validation_probabilities
    ) = transformer_model.predict(
        X_validation_seq
    )

    # ==================================================
    # 8. Validation evaluation
    # ==================================================

    validation_metrics = transformer_model.evaluate(
        y_validation_seq,
        validation_predictions,
        validation_probabilities
    )

    print("\nValidation Metrics:")
    print(validation_metrics)

    # ==================================================
    # 9. Test prediction
    # ==================================================

    (
        test_predictions,
        test_probabilities
    ) = transformer_model.predict(
        X_test_seq
    )

    # ==================================================
    # 10. Test evaluation
    # ==================================================

    test_metrics = transformer_model.evaluate(
        y_test_seq,
        test_predictions,
        test_probabilities
    )

    print("\nTest Metrics:")
    print(test_metrics)