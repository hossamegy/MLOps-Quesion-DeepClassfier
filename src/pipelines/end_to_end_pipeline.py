import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import yaml
import mlflow
import logging

from src.pipelines.preprocessing_pipeline import main as run_preprocessing
from src.models.train import TrainingPipeline
from src.pipelines.evaluation_pipeline import EvaluationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open("config/main_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    experiment_name = config.get("experiment_name", "Arabic_Text_Classification")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="End_to_End_Pipeline") as active_run:
        logger.info(f"Started End-to-End Run with ID: {active_run.info.run_id}")
        
        logger.info("Running Preprocessing Phase...")
        run_preprocessing()
        
        logger.info("Running Training Phase...")
        training_pipeline = TrainingPipeline(config_path="config/main_config.yaml")
        training_pipeline.run()
        
        logger.info("Running Evaluation Phase...")
        eval_pipeline = EvaluationPipeline(config_path="config/main_config.yaml")
        eval_pipeline.evaluate()
        
        logger.info(f"End-to-End Pipeline completed successfully. All artifacts and metrics logged to run {active_run.info.run_id}.")

if __name__ == "__main__":
    main()
