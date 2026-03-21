import pandas as pd

class CsvLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)

    def save_data(self, data: pd.DataFrame):
        data.to_csv(self.file_path, index=False)