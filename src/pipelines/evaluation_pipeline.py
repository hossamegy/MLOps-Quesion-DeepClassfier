import sys
import yaml
import torch
import logging
import mlflow
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.load_data import CsvLoader
from src.features.label_encoder import TargetLabelEncoder
from src.features.pytorch_custom_dataset import ClassificationDataset
from src.models.model import Classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        self.save_path = Path(self.config['data']['model_save_path'])
        self.model_file = self.save_path / "model.pth"
        self.encoder_file = self.save_path / "label_encoder.pkl"
        
        self.encoder = TargetLabelEncoder().load(str(self.encoder_file))
        
    def prepare_data(self):
        logger.info("Preparing data for evaluation...")
        loader = CsvLoader(self.config['data']['processed_v1'])
        df = loader.load_data()
        
        target_col = 'label' if 'label' in df.columns else df.columns[-1]
        text_col = 'question' if 'question' in df.columns else df.columns[0]
        
        labels_encoded = self.encoder.transform(df[target_col].tolist())
        
        dataset = ClassificationDataset(
            texts=df[text_col].tolist(),
            labels=labels_encoded,
            tokenizer=self.tokenizer,
            max_len=self.config['model']['max_length']
        )
        
        self.eval_loader = DataLoader(dataset, batch_size=self.config['model']['batch_size'])
        return target_col

    def evaluate(self):
        target_col = self.prepare_data()
        
        model = Classifier(
            model_name=self.config['model']['name'],
            num_classes=len(self.encoder.classes),
            hidden_dim=self.config['model']['hidden_dim'],
            dropout_rate=self.config['model']['dropout']
        )
        model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        logger.info("Running evaluation...")
        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.encoder.classes)
        
        logger.info(f"Evaluation Accuracy: {acc:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        mlflow.set_experiment(self.config['experiment_name'])
        with mlflow.start_run(run_name="Evaluation"):
            mlflow.log_metric("eval_accuracy", acc)
            report_file = self.save_path / "evaluation_report.txt"
            with open(report_file, "w") as f:
                f.write(report)
            mlflow.log_artifact(str(report_file))
            logger.info("Evaluation metrics logged to MLflow")

if __name__ == "__main__":
    pipeline = EvaluationPipeline(config_path="config/main_config.yaml")
    pipeline.evaluate()
