import pandas as pd
from src.data.preprocessing.base import BasePreprocessor

class DropDuplicates(BasePreprocessor):
    def process(self) -> pd.DataFrame:
        self.df = self.df.drop_duplicates()
        logger.info(f"drop_duplicates applied, new shape: {self.df.shape}")
        return self.df