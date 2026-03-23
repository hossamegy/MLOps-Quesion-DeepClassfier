from pathlib import Path
import mlflow
from sklearn.base import accuracy_score
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, encoder, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.encoder = encoder

    def train(self, target_col, train_loader, val_loader):
    
        mlflow.set_experiment(self.config['experiment_name'])
        
        with mlflow.start_run():
            mlflow.log_params(self.config['model'])
            mlflow.log_param("target_column", target_col)
            
            epochs = self.config['model']['epochs']
            for epoch in range(epochs):
                self.model.train()
                train_losses = []
                
                loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
                for batch in loop:
                    self.optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_losses.append(loss.item())
                    loop.set_postfix(loss=loss.item())
                
                avg_train_loss = sum(train_losses) / len(train_losses)
                
                # Validation
                self.model.eval()
                val_losses = []
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask)
                        loss = self.criterion(outputs, labels)
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
            
            torch.save(self.model.state_dict(), model_file)
            self.encoder.save(str(encoder_file))
            
            mlflow.log_artifact(str(model_file))
            mlflow.log_artifact(str(encoder_file))
            logger.info(f"Model and Encoder saved to {save_path}")