from MarketPulse_AI_project.components.data_transformation import DataTransformation

if __name__ == "__main__":
    raw_data_path = "project_data/raw/nifty50/nifty50_raw.csv"

    transformation = DataTransformation(raw_data_path)

    processed_data_path = transformation.initiate_data_transformation()

    print(f"Transformed data saved at: {processed_data_path}")