from MarketPulse_AI_project.components.feature_engineering import FeatureEngineering
from MarketPulse_AI_project.components.data_transformation import DataTransformation

if __name__ == "__main__":

    raw_data_path = "project_data/raw/nifty50/nifty50_raw.csv"

    data_transformation = DataTransformation(raw_data_path)

    transformed_data_path = data_transformation.initiate_data_transformation()


    feature_engineering =FeatureEngineering(
        transformed_data_path
    )

    feature_data_path = (
        feature_engineering
        .initiate_feature_engineering()
    )

    print(
        f"Feature saved at: {feature_data_path}"
    )