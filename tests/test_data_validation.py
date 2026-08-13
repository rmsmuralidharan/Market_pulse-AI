from MarketPulse_AI_project.components.data_ingestion import DataIngestion
from MarketPulse_AI_project.components.data_validation import DataValidation

if __name__ == "__main__":


    ## stage - 1 Ingestion file path
    data_ingestion = DataIngestion()
    raw_file_path = data_ingestion.initiate_data_ingestion()
    print(f"Raw data saved at: {raw_file_path}")

    ## validation
    data_validation = DataValidation(raw_data_path=raw_file_path)
    validation_artifact = data_validation.initiate_data_validation()

    print(f"Validation status: {validation_artifact.validation_status}")
    print(f"Message: {validation_artifact.message}")
