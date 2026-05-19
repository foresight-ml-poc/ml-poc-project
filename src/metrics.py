"""Foresight metrics: classification metrics computed on hard predictions.

Two callables:
- compute_metrics(y_true, y_pred): Basile contract — used by scripts/main.py
  with hard 0/1 predictions for every model.
- compute_metrics_proba(y_true, y_proba): proba-aware sibling used by
  scripts/train.py and scripts/honest_analysis.py to compute REAL ROC-AUC.
  Not part of the Basile contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute classification metrics on hard 0/1 predictions (Basile contract).

    Returns a dict with EXACTLY these keys: accuracy, precision, recall, f1, roc_auc.
    Every value is a float in [0, 1].

    Notes:
    - ROC-AUC computed on hard 0/1 predictions degenerates toward
      0.5 + correlation/2. Use compute_metrics_proba for real AUC.
    - When y_pred has a single class (e.g. KMeans collapses everything to one
      cluster — unlikely with n_clusters=2 but defensive), roc_auc_score would
      raise; we return 0.5 in that case.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(np.unique(y_pred)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_pred))
    else:
        roc_auc = 0.5

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }


def compute_metrics_proba(y_true: Any, y_proba: Any, threshold: float = 0.5) -> dict[str, float]:
    """Compute metrics using probability scores instead of hard predictions.

    Same return keys as compute_metrics, but roc_auc uses the continuous
    proba scores so it's a real ranking metric (not the degenerate hard-label
    version). Hard-label-derived metrics (accuracy, precision, recall, f1)
    use (proba >= threshold).
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    y_pred = (y_proba >= threshold).astype(int)
    base = compute_metrics(y_true, y_pred)
    base["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return base
