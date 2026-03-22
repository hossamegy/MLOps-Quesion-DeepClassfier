import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Basic schema validation for MLOps pipelines.
    """
    @staticmethod
    def validate_raw_data(df: pd.DataFrame):
        required_columns = ["question", "label"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            if "intent" in df.columns and "label" not in df.columns:
                logger.warning("Found 'intent' but missing 'label'. Renaming 'intent' to 'label'.")
                df.rename(columns={"intent": "label"}, inplace=True)
            else:
                raise ValueError(f"Missing required columns: {missing}")
        
        if df["question"].isnull().any():
            logger.warning("Found null values in 'question' column. These will be handled by the pipeline.")
        
        logger.info("Raw data validation passed.")
        return df

    @staticmethod
    def validate_processed_data(df: pd.DataFrame):
        if df.empty:
            raise ValueError("Processed dataframe is empty!")
        
        if df["question"].str.len().min() == 0:
            logger.warning("Some questions are empty after processing.")
            
        logger.info("Processed data validation passed.")
        return df
