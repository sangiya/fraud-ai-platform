import numpy as np
import pandas as pd
import pytest
from src.fraud_ai_platform.model import FraudDetectionModel


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 500
    n_fraud = int(n * 0.05)
    n_legit = n - n_fraud
    X = pd.DataFrame({
        "amount": np.concatenate([
            np.random.normal(50, 20, n_legit),
            np.random.normal(500, 100, n_fraud),
        ]),
        "hour": np.random.randint(0, 24, n),
        "is_international": np.random.randint(0, 2, n),
        "txn_count_1h": np.random.randint(0, 10, n),
        "txn_count_24h": np.random.randint(0, 30, n),
        "txn_count_7d": np.random.randint(0, 100, n),
        "amount_vs_mean": np.random.uniform(0.5, 15, n),
        "amount_zscore": np.random.uniform(-2, 10, n),
        "is_above_p95": np.random.randint(0, 2, n),
    })
    y = pd.Series(
        np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)
    )
    return X, y


def test_train_returns_metrics(synthetic_data):
    X, y = synthetic_data
    model = FraudDetectionModel()
    metrics = model.train(X, y)
    assert 0 <= metrics.auc_roc <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1 <= 1
    assert 0 < metrics.threshold < 1


def test_predict_proba_range(synthetic_data):
    X, y = synthetic_data
    model = FraudDetectionModel()
    model.train(X, y)
    probs = model.predict_proba(X)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert len(probs) == len(X)


def test_auc_roc_better_than_random(synthetic_data):
    X, y = synthetic_data
    model = FraudDetectionModel()
    metrics = model.train(X, y)
    assert metrics.auc_roc > 0.5


def test_feature_names_preserved(synthetic_data):
    X, y = synthetic_data
    model = FraudDetectionModel()
    model.train(X, y)
    assert model._feature_names == list(X.columns)
