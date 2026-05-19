"""Foresight metrics: classification metrics computed on hard predictions.

Two callables:
- compute_metrics(y_true, y_pred): Basile contract — used by scripts/main.py
  with hard 0/1 predictions for every model.
- compute_metrics_proba(y_true, y_proba): proba-aware sibling used by
  scripts/train.py and scripts/honest_analysis.py to compute REAL ROC-AUC.
  Not part of the Basile contract.

The compute_metrics_proba default threshold is 0.5 (conventional). Task 8's
backtest uses an optimized 0.55 threshold via explicit override (so we can
trade off precision vs net winrate); compute_metrics itself stays at 0.5.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC with safe fallback for degenerate inputs.

    Returns 0.5 (random-classifier AUC) when either y_true OR scores has fewer
    than 2 unique values. sklearn's roc_auc_score raises ValueError otherwise.
    """
    if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute classification metrics on hard 0/1 predictions (Basile contract).

    Returns a dict with EXACTLY these keys: accuracy, precision, recall, f1, roc_auc.
    Every value is a float in [0, 1].

    Notes:
    - ROC-AUC on hard 0/1 predictions degenerates toward 0.5 + correlation/2.
      Use compute_metrics_proba for real AUC.
    - When either y_true or y_pred is single-class (degenerate slice of data
      or model that collapses), roc_auc falls back to 0.5 instead of crashing.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_pred),
    }


def compute_metrics_proba(y_true: Any, y_proba: Any, threshold: float = 0.5) -> dict[str, float]:
    """Compute metrics using probability scores instead of hard predictions.

    Same return keys as compute_metrics, but roc_auc uses the continuous proba
    scores so it's a real ranking metric. Hard-label-derived metrics
    (accuracy/precision/recall/f1) use (proba >= threshold).
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    y_pred = (y_proba >= threshold).astype(int)
    base = compute_metrics(y_true, y_pred)
    base["roc_auc"] = _safe_roc_auc(y_true, y_proba)
    return base
