class RemoveRepeatedWords(BasePreprocessor):
    def process(self) -> pd.DataFrame:
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

        self.df[self.column] = self.df[self.column].apply(_remove_repeats)
        logger.info("remove_repeated_words applied")
        return self.df