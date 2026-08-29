import numpy as np
import pandas as pd
import pytest
from src.fraud_ai_platform.model import FraudDetectionModel
from src.fraud_ai_platform.scoring_engine import ScoringEngine, _risk_level


@pytest.fixture
def trained_engine():
    np.random.seed(42)
    n = 300
    n_fraud = int(n * 0.1)
    n_legit = n - n_fraud
    X = pd.DataFrame({
        "amount": np.concatenate([np.random.normal(50, 10, n_legit), np.random.normal(500, 50, n_fraud)]),
        "hour": np.random.randint(0, 24, n),
        "is_international": np.random.randint(0, 2, n),
        "txn_count_1h": np.random.randint(0, 10, n),
        "txn_count_24h": np.random.randint(0, 30, n),
        "txn_count_7d": np.random.randint(0, 100, n),
        "amount_vs_mean": np.random.uniform(0.5, 15, n),
        "amount_zscore": np.random.uniform(-2, 10, n),
        "is_above_p95": np.random.randint(0, 2, n),
    })
    y = pd.Series(np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int))
    model = FraudDetectionModel()
    model.train(X, y)
    return ScoringEngine(model)


HISTORY = [
    {"card_id": "card-1", "amount": 50.0, "timestamp": "2024-01-01T09:00:00"},
    {"card_id": "card-1", "amount": 55.0, "timestamp": "2024-01-01T09:30:00"},
]
TXN = {"id": "txn-001", "card_id": "card-1", "amount": 60.0, "timestamp": "2024-01-01T10:00:00", "country": "US"}


def test_score_returns_fraud_score(trained_engine):
    score = trained_engine.score(TXN, HISTORY)
    assert score.transaction_id == "txn-001"
    assert 0 <= score.fraud_probability <= 1
    assert score.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert score.latency_ms > 0


def test_risk_levels():
    assert _risk_level(0.1) == "LOW"
    assert _risk_level(0.4) == "MEDIUM"
    assert _risk_level(0.7) == "HIGH"
    assert _risk_level(0.9) == "CRITICAL"


def test_batch_score_length(trained_engine):
    txns = [TXN, {**TXN, "id": "txn-002", "amount": 5000.0}]
    scores = trained_engine.batch_score(txns, HISTORY)
    assert len(scores) == 2
