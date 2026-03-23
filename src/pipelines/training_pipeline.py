import sys
from pathlib import Path

import torch

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import logging
from src.models.train import TrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
if __name__ == "__main__":
    print(torch.cuda.is_available())
    pipeline = TrainingPipeline(config_path="config/main_config.yaml")
    pipeline.run()
