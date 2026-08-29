from src.fraud_ai_platform.features import compute_velocity_features, compute_amount_features, build_feature_vector

HISTORY = [
    {"card_id": "card-1", "amount": 50.0, "timestamp": "2024-01-01T10:00:00"},
    {"card_id": "card-1", "amount": 55.0, "timestamp": "2024-01-01T10:30:00"},
    {"card_id": "card-2", "amount": 200.0, "timestamp": "2024-01-01T09:00:00"},
]
TXN = {"card_id": "card-1", "amount": 80.0, "timestamp": "2024-01-01T11:00:00"}


def test_velocity_counts_same_card_in_window():
    feats = compute_velocity_features(TXN, HISTORY)
    assert feats["txn_count_1h"] == 2


def test_velocity_ignores_other_cards():
    txn2 = {**TXN, "card_id": "card-2"}
    feats2 = compute_velocity_features(txn2, HISTORY)
    assert feats2["txn_count_1h"] == 1


def test_velocity_24h_counts_all_same_card():
    feats = compute_velocity_features(TXN, HISTORY)
    assert feats["txn_count_24h"] >= feats["txn_count_1h"]


def test_amount_zscore_positive_for_high_amount():
    feats = compute_amount_features({"card_id": "card-1", "amount": 5000.0}, HISTORY)
    assert feats["amount_zscore"] > 0


def test_amount_features_empty_history():
    feats = compute_amount_features({"card_id": "new-card", "amount": 100.0}, HISTORY)
    assert feats["amount_vs_mean"] == 0.0
    assert feats["amount_zscore"] == 0.0
    assert feats["is_above_p95"] == 0


def test_build_feature_vector_has_all_keys():
    feats = build_feature_vector(TXN, HISTORY)
    for key in ["amount", "hour", "is_international", "txn_count_1h", "amount_vs_mean"]:
        assert key in feats


def test_international_flag():
    txn_intl = {**TXN, "country": "GB"}
    feats = build_feature_vector(txn_intl, HISTORY)
    assert feats["is_international"] == 1

    txn_domestic = {**TXN, "country": "US"}
    feats_dom = build_feature_vector(txn_domestic, HISTORY)
    assert feats_dom["is_international"] == 0
