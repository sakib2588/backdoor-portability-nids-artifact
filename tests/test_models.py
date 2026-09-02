"""Task 3: LightGBM + MLP NIDS victims with imbalance handling."""
import numpy as np
import pytest

from src.models import (
    train_lightgbm,
    train_mlp,
    mlp_penultimate_features,
    clean_accuracy,
    attack_success_rate,
    predict_labels,
)


def _imbalanced(seed=0, n=8000, d=20, minority=0.02):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < minority).astype(int)          # ~2% botnet, like CTU
    x = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    x[y == 1, :4] += 2.5                                 # a learnable botnet signature
    return x, y


def _botnet_precision_recall(model, x, y):
    pred = predict_labels(model, x)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / int((y == 1).sum()) if (y == 1).sum() else 0.0
    return prec, rec


@pytest.mark.parametrize("kind", ["lgb", "mlp"])
def test_victim_beats_trivial_baseline_and_catches_botnet(kind):
    x, y = _imbalanced(seed=0)
    xte, yte = _imbalanced(seed=1)
    model = train_lightgbm(x, y, seed=0) if kind == "lgb" else train_mlp(x, y, seed=0, epochs=40)
    acc = clean_accuracy(model, xte, yte)
    majority = 1.0 - yte.mean()                          # trivial all-benign baseline
    prec, rec = _botnet_precision_recall(model, xte, yte)
    assert acc >= majority                               # at least as good as trivial
    assert rec > 0.5 and prec > 0.5                      # actually catches botnets (competent NIDS)


def test_mlp_penultimate_features_shape():
    x, y = _imbalanced(seed=0)
    model = train_mlp(x, y, seed=0, epochs=5)
    feats = mlp_penultimate_features(model, x[:500])
    assert feats.shape == (500, 128)


@pytest.mark.parametrize("kind", ["lgb", "mlp"])
def test_no_backdoor_without_poisoning(kind):
    # a clean victim classifies botnet flows as botnet, so "ASR to benign" is low
    x, y = _imbalanced(seed=0)
    xte, yte = _imbalanced(seed=1)
    model = train_lightgbm(x, y, seed=0) if kind == "lgb" else train_mlp(x, y, seed=0, epochs=40)
    botnet = xte[yte == 1]
    asr = attack_success_rate(model, botnet, target=0)   # fraction misclassified benign
    assert asr < 0.5
