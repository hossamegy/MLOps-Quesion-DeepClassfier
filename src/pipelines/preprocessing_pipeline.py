import os
import sys
from pathlib import Path

# Add project root to sys.path at the absolute beginning
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import nltk
from nltk.corpus import stopwords
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
# Use set for faster lookups
arabic_stopwords = set(stopwords.words('arabic'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set default values
    raw_data_path = "experiments/dataset_versions/classfiy_data_v0.csv"
    processed_data_path = "experiments/dataset_versions/processed_data_v1.csv"
    config_path = "config/preprocessing_pipeline.yaml"
    target_column = "question"
    
    logger.info(f"Starting Preprocessing Pipeline")
    loader = CsvLoader(file_path=raw_data_path)
    
    try:
        df = loader.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return

    # Initialize strategies
    preprocessors = {
        "DropNulls": DropNulls(),
        "DropDuplicates": DropDuplicates(),
        "LowerCaser": LowerCaser(),
        "RemoveSpecialCharacters": RemoveSpecialCharacters(),
        "RemoveStopwords": RemoveStopwords(stopwords=list(arabic_stopwords)),
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
            import yaml
            yaml.dump({"pipeline_steps": pipeline_steps}, f)

        mlflow.log_params({"pipeline_steps": str(pipeline_steps), "target_column": target_column})
        mlflow.log_artifact(processed_data_path)
        mlflow.log_artifact(config_path)
        logger.info("Pipeline execution completed and logged to MLflow")

if __name__ == "__main__":
    main()
