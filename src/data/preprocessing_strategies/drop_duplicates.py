import pandas as pd
import logging
from src.data.preprocessing_strategies.base import BasePreprocessor

logger = logging.getLogger(__name__)

class DropDuplicates(BasePreprocessor):
    def process(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df.drop_duplicates().copy()
        logger.info(f"drop_duplicates applied, new shape: {df.shape}")
        return df