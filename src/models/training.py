from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from data.load_data import CsvLoader
from features.label_encoder import TargetLabelEncoder
from features.pytorch_custom_dataset import classificationDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report
from models.model import Classifier

class TrainingPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        self.encoder = TargetLabelEncoder()
        
    def prepare_data(self):
        logger.info("Preparing data for training...")
        loader = CsvLoader(self.config['data']['processed_v1'])
        df = loader.load_data()
        
        target_col = 'label' if 'label' in df.columns else df.columns[-1]
        text_col = 'question' if 'question' in df.columns else df.columns[0]
        
        labels_encoded = self.encoder.fit_transform(df[target_col].tolist())
        
        dataset = ClassificationDataset(
            texts=df[text_col].tolist(),
            labels=labels_encoded,
            tokenizer=self.tokenizer,
            max_len=self.config['model']['max_length']
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['model']['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['model']['batch_size'])
        
        return target_col

    def train(self):
        target_col = self.prepare_data()
        
        model = Classifier(
            model_name=self.config['model']['name'],
            num_classes=len(self.encoder.classes),
            hidden_dim=self.config['model']['hidden_dim'],
            dropout_rate=self.config['model']['dropout']
        )
        model.to(self.device)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config['model']['learning_rate']))
        
        mlflow.set_experiment(self.config['experiment_name'])
        
        with mlflow.start_run():
            mlflow.log_params(self.config['model'])
            mlflow.log_param("target_column", target_col)
            
            epochs = self.config['model']['epochs']
            for epoch in range(epochs):
                model.train()
                train_losses = []
                
                loop = tqdm(self.train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
                for batch in loop:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    loop.set_postfix(loss=loss.item())
                
                avg_train_loss = sum(train_losses) / len(train_losses)
                
                # Validation
                model.eval()
                val_losses = []
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch in self.val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                        val_losses.append(loss.item())
                        
                        preds = torch.argmax(outputs, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_acc = accuracy_score(all_labels, all_preds)
                
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Save Model and Encoder
            save_path = Path(self.config['data']['model_save_path'])
            save_path.mkdir(parents=True, exist_ok=True)
            
            model_file = save_path / "model.pth"
            encoder_file = save_path / "label_encoder.pkl"
            
            torch.save(model.state_dict(), model_file)
            self.encoder.save(str(encoder_file))
            
            mlflow.log_artifact(str(model_file))
            mlflow.log_artifact(str(encoder_file))
            logger.info(f"Model and Encoder saved to {save_path}")