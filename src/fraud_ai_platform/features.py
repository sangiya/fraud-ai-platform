from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np


def compute_velocity_features(transaction: dict, history: list[dict]) -> dict:
    """Count transactions from same card in the last 1h, 24h, and 7d windows."""
    card = transaction.get("card_id", "")
    ts = datetime.fromisoformat(str(transaction.get("timestamp", datetime.utcnow().isoformat())))
    windows = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
    }
    result = {}
    for name, window in windows.items():
        cutoff = ts - window
        count = sum(
            1
            for h in history
            if h.get("card_id") == card
            and datetime.fromisoformat(str(h.get("timestamp", ts.isoformat()))) >= cutoff
        )
        result[f"txn_count_{name}"] = count
    return result


def compute_amount_features(transaction: dict, history: list[dict]) -> dict:
    amounts = [
        float(h.get("amount", 0))
        for h in history
        if h.get("card_id") == transaction.get("card_id")
    ]
    amount = float(transaction.get("amount", 0))
    if not amounts:
        return {"amount_vs_mean": 0.0, "amount_zscore": 0.0, "is_above_p95": 0}
    mean = float(np.mean(amounts))
    std = float(np.std(amounts)) or 1.0
    p95 = float(np.percentile(amounts, 95))
    return {
        "amount_vs_mean": amount / mean if mean else 0.0,
        "amount_zscore": (amount - mean) / std,
        "is_above_p95": int(amount > p95),
    }


def build_feature_vector(transaction: dict, history: list[dict]) -> dict:
    feats: dict[str, float | int] = {}
    feats["amount"] = float(transaction.get("amount", 0))
    ts_str = str(transaction.get("timestamp", datetime.utcnow().isoformat()))
    feats["hour"] = datetime.fromisoformat(ts_str).hour
    feats["is_international"] = int(transaction.get("country", "US") != "US")
    feats.update(compute_velocity_features(transaction, history))
    feats.update(compute_amount_features(transaction, history))
    return feats
