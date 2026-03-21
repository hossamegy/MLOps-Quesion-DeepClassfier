import pandas as pd
import logging
from typing import List, Optional
from src.data.preprocessing_strategies.base import BasePreprocessor

logger = logging.getLogger(__name__)

class RemoveStopwords(BasePreprocessor):
    def __init__(self, stopwords: Optional[List[str]] = None):
        self.stopwords = set(stopwords) if stopwords else set()

    def process(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df.copy()
        df[column] = df[column].apply(
            lambda x: ' '.join([w for w in x.split() if w not in self.stopwords])
            if isinstance(x, str) else x
        )
        logger.info("remove_stopwords applied")
        return df