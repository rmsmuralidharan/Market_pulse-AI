from MarketPulse_AI_project.components.data_preprocessing import DataPreprocessing

if __name__ == "__main__":

    train_path = "project_data/train/train.csv"
    validation_path = "project_data/validation/validation.csv"
    test_path = "project_data/test/test.csv"

    preprocessing = DataPreprocessing(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path
    )

    (
        x_train_scaled,
        x_validation_scaled,
        x_test_scaled,
        y_train,
        y_validation,
        y_test,
        scaler
    ) = preprocessing.initiate_data_preprocessing()

    print("\nTrain shape:")
    print(x_train_scaled.shape)

    print("\nValidation shape:")
    print(x_validation_scaled.shape)

    print("\nTest shape:")
    print(x_test_scaled.shape)

    print("\nTrain target shape:")
    print(y_train.shape)

    print("\nValidation target shape:")
    print(y_validation.shape)

    print("\nTest target shape:")
    print(y_test.shape)

    print("\nFirst scaled training row:")
    print(x_train_scaled[0])

