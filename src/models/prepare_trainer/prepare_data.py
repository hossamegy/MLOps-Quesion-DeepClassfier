import logging
from torch.utils.data import DataLoader, random_split

from src.data.load_data import CsvLoader
from src.features.pytorch_custom_dataset import ClassificationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrepareData:
    def __init__(self, config: str, encoder, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.encoder = encoder
        
    def prepare(self):
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
        
        train_size = int(self.config['model']['train_val_split'] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['model']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['model']['batch_size'])
        
        return target_col, train_loader, val_loader