from pathlib import Path
import mlflow
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
import logging
import yaml

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config, encoder, tokenizer, model, optimizer, criterion, device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.encoder = encoder
        self.tokenizer = tokenizer
        
        if (self.config['model']['fine_tune']):
            self.model.load_state_dict(torch.load(self.config['model']['head_model']))

        logger.info(f"device: {self.device}")

    def train(self, target_col, train_loader, val_loader):

        mlflow.set_experiment(self.config['experiment_name'])

        with mlflow.start_run():
            mlflow.log_params(self.config['model'])
            mlflow.log_param("target_column", target_col)

            epochs = self.config['model']['epochs']
            best_val_acc = 0.0
            best_epoch = -1

            for epoch in range(epochs):
                # TRAIN =================
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

                # VALIDATION =================
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

                # LOG =================
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                logger.info(
                    f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

                # SAVE BEST MODEL =================
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1

                    save_path = Path(self.config['data']['model_save_path'])
                    save_path.mkdir(parents=True, exist_ok=True)

                    model_dir = save_path / "bert_model"
                    model_dir.mkdir(parents=True, exist_ok=True)

                    encoder_file = save_path / "label_encoder.pkl"
                    full_model_file = save_path / "full_model.pth"
                    config_file = save_path / "config.yaml"

                    # Save BERT + tokenizer
                    self.model.bert.save_pretrained(model_dir)
                    self.tokenizer.save_pretrained(model_dir)

                    # Save classifier
                    torch.save(self.model.state_dict(), full_model_file)

                    # Save encoder
                    self.encoder.save(str(encoder_file))

                    # Save config (reproducibility)
                    with open(config_file, "w") as f:
                        yaml.dump(self.config, f)

                    # MLflow logging
                    mlflow.log_artifacts(str(model_dir), artifact_path="bert_model")
                    mlflow.log_artifact(str(full_model_file))
                    mlflow.log_artifact(str(encoder_file))
                    mlflow.log_artifact(str(config_file))

                    logger.info(
                        f"Best model saved at epoch {best_epoch} "
                        f"with acc {best_val_acc:.4f}"
                    )

            # FINAL LOG =================
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.log_param("best_epoch", best_epoch)

            if best_val_acc == 0:
                logger.warning("No improvement during training")
    