import pandas as pd
from src.data.preprocessing.base import BasePreprocessor

class LowerCaser(BasePreprocessor):
    def process(self) -> pd.DataFrame:
        self.df[self.column] = self.df[self.column].str.lower()
        return self.df