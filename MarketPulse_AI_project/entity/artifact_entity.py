from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str