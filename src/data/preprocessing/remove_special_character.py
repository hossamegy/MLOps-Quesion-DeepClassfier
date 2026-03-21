import pandas as pd
from src.data.preprocessing.base import BasePreprocessor

class RemoveSpecialCharacters(BasePreprocessor):
    def process(self) -> pd.DataFrame:
        self.df[self.column] = self.df[self.column].apply(
            lambda x: re.sub(r'[^\w\sء-ي]', '', x) if isinstance(x, str) else x
        )
        logger.info("remove_special_characters applied")
        return self.df