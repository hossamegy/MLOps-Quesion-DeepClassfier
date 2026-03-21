import pandas as pd
import logging
from src.data.preprocessing_strategies.base import BasePreprocessor

logger = logging.getLogger(__name__)

class DropNulls(BasePreprocessor):
    def process(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df.dropna(subset=[column]).copy()
        logger.info(f"drop_nulls applied, new shape: {df.shape}")
        return df