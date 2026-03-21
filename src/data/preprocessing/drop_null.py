import pandas as pd
from src.data.preprocessing.base import BasePreprocessor

class DropNulls(BasePreprocessor):
    def process(self) -> pd.DataFrame:
        self.df = self.df.dropna(subset=[self.column])
        logger.info(f"drop_nulls applied, new shape: {self.df.shape}")
        return self.df