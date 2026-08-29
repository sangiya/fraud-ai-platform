# fraud-ai-platform

ML fraud detection pipeline: velocity features, Random Forest scoring, threshold optimization, and FastAPI serving.

## Features

- **Feature Engineering** — velocity counts (1h/24h/7d), amount Z-score, international flag, percentile comparison
- **Fraud Detection Model** — Random Forest with `class_weight="balanced"`, threshold optimised for ≥90% precision
- **Scoring Engine** — real-time single-transaction scoring with risk level bucketing (LOW/MEDIUM/HIGH/CRITICAL) and latency tracking
- **Pipeline** — end-to-end training pipeline from raw transaction dicts to trained model and metrics

## Structure

```
src/fraud_ai_platform/
    features.py          # Velocity, amount, and behavioural feature extraction
    model.py             # RandomForest with ROC-AUC + threshold optimisation
    scoring_engine.py    # Real-time fraud scoring with risk level assignment
    pipeline.py          # End-to-end training pipeline
tests/
    test_features.py
    test_model.py
    test_scoring_engine.py
```

## Usage

```python
from fraud_ai_platform.pipeline import FraudPipeline
from fraud_ai_platform.scoring_engine import ScoringEngine

history = [
    {"card_id": "card-1", "amount": 50.0, "timestamp": "2024-01-01T09:00:00"},
]

# Training
pipeline = FraudPipeline()
transactions = [{"card_id": "card-1", "amount": 500.0, "timestamp": "2024-01-01T10:00:00", "country": "US"}]
labels = [1]  # 1 = fraud, 0 = legitimate
metrics = pipeline.train(transactions * 50, history, labels * 50)
print(f"AUC-ROC: {metrics.auc_roc:.3f}, Threshold: {metrics.threshold:.2f}")

# Scoring
engine = ScoringEngine(pipeline.model)
score = engine.score(transactions[0], history)
print(f"Fraud probability: {score.fraud_probability:.3f} | Risk: {score.risk_level}")
```

## Metrics

Training reports: AUC-ROC, precision, recall, F1, and the optimal probability threshold that maximises recall while maintaining ≥90% precision.

## Running Tests

```bash
pip install -r requirements.txt
pytest --tb=short
```
