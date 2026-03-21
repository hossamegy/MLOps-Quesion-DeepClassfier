class RemoveStopwords(BasePreprocessor):
    def __init__(self, df: pd.DataFrame, column: str, stopwords: Optional[List[str]] = None):
        super().__init__(df, column)
        self.stopwords = stopwords if stopwords else []

    def process(self) -> pd.DataFrame:
        self.df[self.column] = self.df[self.column].apply(
            lambda x: ' '.join([w for w in x.split() if w not in self.stopwords])
            if isinstance(x, str) else x
        )
        logger.info("remove_stopwords applied")
        return self.df