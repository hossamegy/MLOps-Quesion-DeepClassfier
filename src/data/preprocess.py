import pandas as pd
import logging
import os
from typing import Optional, List
from src.data.preprocessing_strategies.base import BasePreprocessor
from src.data.validation import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(
            self,
            preprocess_dict: dict[str, BasePreprocessor],
            df: pd.DataFrame,
            column: str,
            stopwords: Optional[List[str]] = None
        ):
        """
        Args:
            preprocess_dict (dict[str, BasePreprocessor]): Dictionary of preprocessors to apply
            df (pd.DataFrame): DataFrame to preprocess
            column (str): Target text column
            stopwords (Optional[List[str]]): List of stopwords
        """
        self.df = df.copy()
        self.column = column
        self.stopwords = stopwords if stopwords else []
        self.preprocess_dict = preprocess_dict

    def run(self) -> pd.DataFrame:
        if self.column not in self.df.columns:
            raise ValueError(f"Target column '{self.column}' not found in DataFrame.")

        logger.info(f"Initial shape: {self.df.shape}")
        for name, preprocessor in self.preprocess_dict.items():
            logger.info(f"Applying {name} preprocessor")
            self.df = preprocessor.process(df=self.df, column=self.column)
            
        self.df = DataValidator.validate_processed_data(self.df)
        logger.info(f"Final shape: {self.df.shape}")
        return self.df

    def save_data(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.df.to_csv(file_path, index=False)
        logger.info(f"Saved processed data to {file_path}")

    def get_preprocessing_pipeline(self) -> list[str]:
        return list(self.preprocess_dict.keys())