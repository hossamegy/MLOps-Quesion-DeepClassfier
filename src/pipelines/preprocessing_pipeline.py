import os
from nltk.corpus import stopwords
import nltk

import mlflow
import logging

from src.data.load_data import CsvLoader
from src.data.preprocess import PreprocessingPipeline
from src.data.preprocessing_strategies.lower_case import LowerCaser
from src.data.preprocessing_strategies.remove_special_character import RemoveSpecialCharacters
from src.data.preprocessing_strategies.remove_stopwords import RemoveStopwords
from src.data.preprocessing_strategies.drop_null import DropNulls
from src.data.preprocessing_strategies.drop_duplicates import DropDuplicates
from src.data.preprocessing_strategies.remove_repeated_words import RemoveRepeatedWords


nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    raw_data_path = "data/raw/train.csv"
    processed_data_path = "data/processed/train.csv"
    config_path = "config/preprocessing_pipeline.yaml"
    target_column = "question"
    
  
    logger.info(f"Starting Preprocessing Pipeline")
    loader = CsvLoader(file_path=raw_data_path)
    
    try:
        df = loader.load_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    preprocessors = {
        "DropNulls": DropNulls(),
        "DropDuplicates": DropDuplicates(),
        "LowerCaser": LowerCaser(),
        "RemoveSpecialCharacters": RemoveSpecialCharacters(),
        "RemoveStopwords": RemoveStopwords(stopwords=arabic_stopwords),
        "RemoveRepeatedWords": RemoveRepeatedWords()
    }

    preprocessor = PreprocessingPipeline(
        preprocess_dict=preprocessors,
        df=df,
        column=target_column,
        stopwords=arabic_stopwords
    )

    # MLflow Tracking
    mlflow.set_experiment("Data_Preprocessing")
    with mlflow.start_run():
        preprocessor.run()
        preprocessor.save_data(file_path=processed_data_path)
        
        pipeline_steps = preprocessor.get_preprocessing_pipeline()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
        mlflow.log_params({"pipeline_steps": str(pipeline_steps), "target_column": target_column})
        mlflow.log_artifact(config_path)
        mlflow.log_artifact(processed_data_path)
        logger.info("Pipeline execution completed and logged to MLflow")

if __name__ == "__main__":
    main()
