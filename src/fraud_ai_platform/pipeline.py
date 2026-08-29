from __future__ import annotations
import pandas as pd
import numpy as np
from .features import build_feature_vector
from .model import FraudDetectionModel, TrainingMetrics


class FraudPipeline:
    """End-to-end fraud detection pipeline: feature engineering + model training + evaluation."""

    def __init__(self):
        self._model = FraudDetectionModel()

    def build_dataset(
        self, transactions: list[dict], history: list[dict], labels: list[int]
    ) -> tuple[pd.DataFrame, pd.Series]:
        rows = [build_feature_vector(t, history) for t in transactions]
        X = pd.DataFrame(rows)
        y = pd.Series(labels, dtype=int)
        return X, y

    def train(
        self, transactions: list[dict], history: list[dict], labels: list[int]
    ) -> TrainingMetrics:
        X, y = self.build_dataset(transactions, history, labels)
        return self._model.train(X, y)

    def predict_proba(self, transactions: list[dict], history: list[dict]) -> np.ndarray:
        rows = [build_feature_vector(t, history) for t in transactions]
        X = pd.DataFrame(rows)
        return self._model.predict_proba(X)

    @property
    def model(self) -> FraudDetectionModel:
        return self._model
