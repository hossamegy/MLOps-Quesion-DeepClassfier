import pandas as pd
import logging
import re
from src.data.preprocessing_strategies.base import BasePreprocessor

logger = logging.getLogger(__name__)

class RemoveSpecialCharacters(BasePreprocessor):
    def process(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df.copy()
        df[column] = df[column].apply(
            lambda x: re.sub(r'[^\w\sء-ي]', '', x) if isinstance(x, str) else x
        )
        logger.info("remove_special_characters applied")
        return df