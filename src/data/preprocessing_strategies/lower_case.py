import pandas as pd
import logging
from src.data.preprocessing_strategies.base import BasePreprocessor

logger = logging.getLogger(__name__)

class LowerCaser(BasePreprocessor):
    def process(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df.copy()
        df[column] = df[column].str.lower()
        logger.info(f"lower_case applied, new shape: {df.shape}")
        return df