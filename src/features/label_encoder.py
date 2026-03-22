import pickle
from sklearn.preprocessing import LabelEncoder
from typing import List

class TargetLabelEncoder:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, target: List[str]):
        self.encoder.fit(target)
        self.is_fitted = True
        return self

    def transform(self, target: List[str]):
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before calling transform")
        return self.encoder.transform(target)

    def fit_transform(self, target: List[str]):
        self.is_fitted = True
        return self.encoder.fit_transform(target)

    def inverse_transform(self, target_ids):
        return self.encoder.inverse_transform(target_ids)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.encoder, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.encoder = pickle.load(f)
        self.is_fitted = True
        return self