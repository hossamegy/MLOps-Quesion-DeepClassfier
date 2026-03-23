from pathlib import Path
import torch
from transformers import AutoTokenizer
import yaml
from src.features.label_encoder import TargetLabelEncoder
from src.models.classifier_model import Classifier
from src.models.prepare_trainer.prepare_data import PrepareData
from src.models.prepare_trainer.trainer import Trainer


class TrainingPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.save_path = Path(self.config['data']['model_save_path'])
        self.model_file = self.save_path / "model.pth"
        self.encoder_file = self.save_path / "label_encoder.pkl"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['tokenizer'])
        
        self.encoder = TargetLabelEncoder().load(str(self.encoder_file))

        self.model = Classifier(
            model_name=self.config['model']['name'],
            num_classes=self.config['model']['num_classes'],
            hidden_dim=self.config['model']['hidden_dim'],
            dropout_rate=self.config['model']['dropout']
        )
        self.model.to(self.device)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config['model']['learning_rate']))
      

        self.prepare_data = PrepareData(
            self.config,
            self.encoder,
            self.tokenizer,
        )
        
        self.trainer = Trainer(
            self.config,
            self.encoder,
            self.model,
            self.optimizer,
            self.criterion,
            self.device,
        )

    def run(self):
        target_col, train_loader, val_loader = self.prepare_data.prepare()
        self.trainer.train(
            target_col, 
            train_loader, 
            val_loader
        )

