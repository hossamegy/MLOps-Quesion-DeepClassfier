import pandas as pd
from typing import List, Optional
from src.data.preprocessing.base import BasePreprocessor

class PreprocessingPipeline:
    def __init__(
            self, preprocess_list: List[BasePreprocessor],
            df: pd.DataFrame, 
            column: str, 
            stopwords: Optional[List[str]] = None
        ):

        """
        Args:
            preprocess_list (List[BasePreprocessor]): List of preprocessors to apply
            df (pd.DataFrame): DataFrame to preprocess
            column (str): Column to preprocess
            stopwords (Optional[List[str]]): List of stopwords to remove
        """

        self.df = df
        self.column = column
        self.stopwords = stopwords
        self.preprocess_list = preprocess_list

    def run(self) -> pd.DataFrame:
        for preprocessor in self.preprocess_list:
            self.df = preprocessor.process()
        return self.df