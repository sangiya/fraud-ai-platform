from __future__ import annotations
import time
from dataclasses import dataclass
import pandas as pd
from .features import build_feature_vector
from .model import FraudDetectionModel


@dataclass
class FraudScore:
    transaction_id: str
    fraud_probability: float
    risk_level: str
    latency_ms: float
    threshold_used: float


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    if prob < 0.6:
        return "MEDIUM"
    if prob < 0.85:
        return "HIGH"
    return "CRITICAL"


class ScoringEngine:
    def __init__(self, model: FraudDetectionModel):
        self._model = model

    def score(self, transaction: dict, history: list[dict]) -> FraudScore:
        start = time.perf_counter()
        features = build_feature_vector(transaction, history)
        X = pd.DataFrame([features])
        prob = float(self._model.predict_proba(X)[0])
        latency = (time.perf_counter() - start) * 1000
        return FraudScore(
            transaction_id=str(transaction.get("id", "unknown")),
            fraud_probability=prob,
            risk_level=_risk_level(prob),
            latency_ms=latency,
            threshold_used=self._model._threshold,
        )

    def batch_score(self, transactions: list[dict], history: list[dict]) -> list[FraudScore]:
        return [self.score(t, history) for t in transactions]
