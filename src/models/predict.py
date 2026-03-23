import torch
from transformers import AutoModel, AutoTokenizer

from src.models.classifier_model import Classifier

model_dir = ""

tokenizer = AutoTokenizer.from_pretrained(model_dir)
bert = AutoModel.from_pretrained(model_dir)

model = Classifier(model_name=model_dir, num_classes=10)
model.load_state_dict(torch.load("full_model.pth"))

model.eval()