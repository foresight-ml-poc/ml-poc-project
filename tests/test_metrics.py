"""Contract tests for src/metrics.py."""
import importlib.util
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_ROOT / "src" / "metrics.py"


def _metrics():
    spec = importlib.util.spec_from_file_location("project_metrics", METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_returns_dict_with_required_keys():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    result = _metrics().compute_metrics(y_true, y_pred)
    assert isinstance(result, dict)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in result, f"missing key: {key}"
        assert isinstance(result[key], float), f"value for {key} is not a float"


def test_all_correct_gives_perfect_metrics():
    y = np.array([0, 1, 1, 0, 1, 0])
    result = _metrics().compute_metrics(y, y)
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_compute_metrics_proba_returns_real_auc():
    """proba-based AUC must differ from hard-label AUC when probabilities are informative."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 1000)
    # Informative probabilities: y_true=1 → mean 0.7, y_true=0 → mean 0.3, with noise
    y_proba = np.where(y_true == 1, 0.7, 0.3) + rng.normal(0, 0.15, 1000)
    y_proba = np.clip(y_proba, 0.0, 1.0)
    result = _metrics().compute_metrics_proba(y_true, y_proba)
    # AUC should be high (~0.9+) for these informative probas
    assert result["roc_auc"] > 0.8, f"AUC {result['roc_auc']} too low for informative probas"
    # Hard 0/1 version would lose this nuance
    hard = _metrics().compute_metrics(y_true, (y_proba >= 0.5).astype(int))
    assert result["roc_auc"] > hard["roc_auc"], "proba AUC should exceed hard AUC for these probas"


def test_compute_metrics_handles_single_class_predictions():
    """When y_pred is all 0 or all 1, roc_auc on hard labels falls back to 0.5 (not crash)."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred_all_zero = np.zeros_like(y_true)
    result = _metrics().compute_metrics(y_true, y_pred_all_zero)
    assert result["roc_auc"] == 0.5
    assert result["accuracy"] == 0.5  # half are 0


def test_metric_values_in_unit_interval():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 500)
    y_pred = rng.integers(0, 2, 500)
    result = _metrics().compute_metrics(y_true, y_pred)
    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} outside [0, 1]"


def test_compute_metrics_handles_single_class_y_true():
    """y_true entirely 0 or entirely 1 → roc_auc=0.5 fallback, no ValueError."""
    y_true = np.zeros(20, dtype=int)
    y_pred = np.array([0, 1] * 10)
    result = _metrics().compute_metrics(y_true, y_pred)
    assert result["roc_auc"] == 0.5

    # Same for the proba variant
    y_proba = np.linspace(0.1, 0.9, 20)
    result_proba = _metrics().compute_metrics_proba(y_true, y_proba)
    assert result_proba["roc_auc"] == 0.5
