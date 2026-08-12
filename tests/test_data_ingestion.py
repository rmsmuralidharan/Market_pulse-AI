from MarketPulse_AI_project.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    data_ingestion = DataIngestion()

    raw_file_path = data_ingestion.initiate_data_ingestion()

    print(f"Raw data saved at: {raw_file_path}")