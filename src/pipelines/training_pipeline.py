import sys
from pathlib import Path

# Add project root to sys.path at the very beginning
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import logging
from src.models.training import TrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
if __name__ == "__main__":
    pipeline = TrainingPipeline(config_path="config/main_config.yaml")
    pipeline.run()
