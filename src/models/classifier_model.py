from torch import nn
from transformers import AutoModel

class Classifier(nn.Module):
    """
    BERT-based Classifier with custom head.
    """
    def __init__(self, model_name: str, num_classes: int, hidden_dim: int = 15, dropout_rate: float = 0.3):
        super(Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Using the [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        x = self.dropout(cls_embedding)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x