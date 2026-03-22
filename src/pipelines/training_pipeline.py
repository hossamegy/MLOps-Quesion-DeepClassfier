import sys

import logging
from pathlib import Path
from src.pipelines.training_pipeline import TrainingPipeline

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
if __name__ == "__main__":
    pipeline = TrainingPipeline(config_path="config/main_config.yaml")
    pipeline.train()
