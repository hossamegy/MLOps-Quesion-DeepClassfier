import pickle
import os
import logging
from sklearn.preprocessing import LabelEncoder
from typing import List, Any

logger = logging.getLogger(__name__)

class TargetLabelEncoder:
    """
    A wrapper around sklearn's LabelEncoder with persistence capabilities.
    """
    def __init__(self):
        self.encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, target: List[str]):
        logger.info("Fitting LabelEncoder...")
        self.encoder.fit(target)
        self.is_fitted = True
        return self

    def transform(self, target: List[str]) -> Any:
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before calling transform")
        return self.encoder.transform(target)

    def fit_transform(self, target: List[str]) -> Any:
        self.is_fitted = True
        return self.encoder.fit_transform(target)

    def inverse_transform(self, target_ids: List[int]) -> List[str]:
        return self.encoder.inverse_transform(target_ids)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.encoder, f)
        logger.info(f"LabelEncoder saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            logger.info(f"Create LabelEncoder and saved to {path}")
            self.encoder = LabelEncoder()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.encoder, f)  

        with open(path, "rb") as f:
            self.encoder = pickle.load(f)
        self.is_fitted = True
        logger.info(f"LabelEncoder loaded from {path}")
        return self

    @property
    def classes(self):
        return self.encoder.classes_