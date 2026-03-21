import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class CsvLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at {self.file_path}")
        logger.info(f"Loading data from {self.file_path}")
        return pd.read_csv(self.file_path)

    def save_data(self, data: pd.DataFrame):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        data.to_csv(self.file_path, index=False)
        logger.info(f"Saved data to {self.file_path}")