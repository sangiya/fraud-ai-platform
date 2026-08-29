from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


@dataclass
class TrainingMetrics:
    auc_roc: float
    precision: float
    recall: float
    f1: float
    threshold: float


class FraudDetectionModel:
    def __init__(self):
        self._model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._threshold: float = 0.5
        self._feature_names: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> TrainingMetrics:
        self._feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        self._model.fit(X_train, y_train)
        probs = self._model.predict_proba(X_test)[:, 1]
        self._threshold = self._find_optimal_threshold(y_test.values, probs, target_precision=0.9)
        preds = (probs >= self._threshold).astype(int)
        return TrainingMetrics(
            auc_roc=float(roc_auc_score(y_test, probs)),
            precision=float(precision_score(y_test, preds, zero_division=0)),
            recall=float(recall_score(y_test, preds, zero_division=0)),
            f1=float(f1_score(y_test, preds, zero_division=0)),
            threshold=self._threshold,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X[self._feature_names])[:, 1]

    def _find_optimal_threshold(
        self, y_true: np.ndarray, probs: np.ndarray, target_precision: float = 0.9
    ) -> float:
        best_threshold = 0.5
        best_recall = 0.0
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (probs >= t).astype(int)
            p = precision_score(y_true, preds, zero_division=0)
            r = recall_score(y_true, preds, zero_division=0)
            if p >= target_precision and r > best_recall:
                best_recall = r
                best_threshold = float(t)
        return best_threshold
