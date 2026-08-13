from MarketPulse_AI_project.components.data_split import DataSplit

if __name__ == "__main__":
    featured_data_path = "project_data/processed/nifty50/nifty50_features.csv"

    split = DataSplit()

    train_path, validation_path, test_path = split.initiate_data_split(
        featured_data_path
    )

    print(f"Train data saved at: {train_path}")
    print(f"Validation data saved at: {validation_path}")
    print(f"Test data saved at: {test_path}")