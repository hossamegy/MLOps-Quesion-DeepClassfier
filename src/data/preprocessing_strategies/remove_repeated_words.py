import pandas as pd
import logging
from src.data.preprocessing_strategies.base import BasePreprocessor

logger = logging.getLogger(__name__)

class RemoveRepeatedWords(BasePreprocessor):
    def process(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        def _remove_repeats(text: str) -> str:
            if not isinstance(text, str):
                return text
            seen = set()
            result = []
            for w in text.split():
                if w not in seen:
                    seen.add(w)
                    result.append(w)
            return ' '.join(result)

        df = df.copy()
        df[column] = df[column].apply(_remove_repeats)
        logger.info("remove_repeated_words applied")
        return df