import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePreprocessor:
    def process(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Override in subclass"""
        raise NotImplementedError