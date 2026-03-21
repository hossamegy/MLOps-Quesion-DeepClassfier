import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePreprocessor:
    def __init__(self, df: pd.DataFrame, column: str):
        self.df = df.copy()
        self.column = column

    def process(self) -> pd.DataFrame:
        """Override in subclass"""
        raise NotImplementedError