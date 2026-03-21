from src.data.load_data import CsvLoader
from src.data.preprocess import PreprocessingPipeline
from src.data.preprocessing.lower_case import LowerCaser
from src.data.preprocessing.remove_special_character import RemoveSpecialCharacters
from src.data.preprocessing.remove_stopwords import RemoveStopwords
from src.data.preprocessing.drop_null import DropNulls
from src.data.preprocessing.drop_duplicates import DropDuplicates
from src.data.preprocessing.remove_repeated_words import RemoveRepeatedWords

loader = CsvLoader(file_path="data/raw/train.csv")
preprocessor = PreprocessingPipeline(
    preprocess_list=[
        DropNulls(),
        DropDuplicates(),
        LowerCaser(),
        RemoveSpecialCharacters(),
        RemoveStopwords(stopwords=self.stopwords),
        RemoveRepeatedWords()
    ],
    df=loader.load(),
    column="question"
)

if __name__ == "__main__":
    preprocessor.run()
    loader.save_data(preprocessor.df)

