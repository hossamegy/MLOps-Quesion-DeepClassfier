import sys
from pathlib import Path
import os
import nltk
from nltk.corpus import stopwords
import mlflow
import logging
import yaml

nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.load_data import CsvLoader
from src.data.preprocess import PreprocessingPipeline
from src.data.preprocessing_strategies.lower_case import LowerCaser
from src.data.preprocessing_strategies.remove_special_character import RemoveSpecialCharacters
from src.data.preprocessing_strategies.remove_stopwords import RemoveStopwords
from src.data.preprocessing_strategies.drop_null import DropNulls
from src.data.preprocessing_strategies.drop_duplicates import DropDuplicates
from src.data.preprocessing_strategies.remove_repeated_words import RemoveRepeatedWords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open("config/main_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set default values
    raw_data_path = config['data']['raw_path']
    processed_data_path = config['data']['processed']
    config_path = "config/preprocessing_pipeline.yaml"
    target_column = config['data']['target']
    
    logger.info(f"Starting Preprocessing Pipeline")
    loader = CsvLoader(file_path=raw_data_path)
    
    try:
        df = loader.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return

    preprocessors = {
        "DropNulls": DropNulls(),
        "DropDuplicates": DropDuplicates(),
        "LowerCaser": LowerCaser(),
        "RemoveSpecialCharacters": RemoveSpecialCharacters(),
        # "RemoveStopwords": RemoveStopwords(stopwords=list(arabic_stopwords)),
        "RemoveRepeatedWords": RemoveRepeatedWords()
    }

    preprocessor = PreprocessingPipeline(
        preprocess_dict=preprocessors,
        df=df,
        column=target_column,
        stopwords=list(arabic_stopwords)
    )

    # MLflow Tracking
    mlflow.set_experiment("Data_Preprocessing")
    with mlflow.start_run():
        preprocessor.run()
        preprocessor.save_data(file_path=processed_data_path)
        
        # Save config
        pipeline_steps = preprocessor.get_preprocessing_pipeline() 
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump({"pipeline_steps": pipeline_steps}, f)

        mlflow.log_params({"pipeline_steps": str(pipeline_steps), "target_column": target_column})
        mlflow.log_artifact(processed_data_path)
        mlflow.log_artifact(config_path)
        logger.info("Pipeline execution completed and logged to MLflow")

if __name__ == "__main__":
    main()
