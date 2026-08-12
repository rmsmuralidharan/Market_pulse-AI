from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_dir: str

    symbol: str

    def __init__(self, raw_data_dir: str, symbol: str):
        self.raw_data_dir = raw_data_dir
        self.symbol = symbol



@dataclass
class DataValidationConfig:
    required_columns: list