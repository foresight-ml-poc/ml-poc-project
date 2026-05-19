"""Foresight dataset loader.

Loads the 3000-signal CSV + 144-pt trajectories, computes the binary target
at the reference horizon (60 min), applies the anti-leakage allowlist (only
the 26 named FEATURES enter X), and returns a walk-forward split (last 20 %
is test).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    BUCKETS, FEATURES, HORIZON_REF_IDX, OUTPUT_COLUMNS, PATHS_DIR,
    PROCESSED_DATA_DIR, RAW_DATA_DIR, TARGET_COLUMN,
)


def _read_trajectory(signal_id: str) -> np.ndarray:
    path = PATHS_DIR / f"{signal_id}.json"
    return np.asarray(json.loads(path.read_text())["price"], dtype=float)


def _compute_target(df: pd.DataFrame, horizon_idx: int) -> pd.Series:
    """Compute direction_correct@horizon as a binary target.

    Trajectory index convention (see scripts/generate_dataset.py _simulate_trajectory):
        trajectory[i] = price AFTER step (i+1) has elapsed
                      = price at t = (i+1) * STEP_MIN minutes
        trajectory[0]    -> t =   10 min
        trajectory[5]    -> t =   60 min  (HORIZON_REF_IDX=6 → Branch A target)
        trajectory[143]  -> t = 1440 min  (24 h)
    price_start (the t=0 price) is stored separately in market_price_at_signal,
    NOT as trajectory[0].

    So for horizon_idx h (1-indexed, like HORIZON_REF_IDX), we read
    trajectory[h - 1], which is the price at t = h * STEP_MIN minutes after the signal.

    Target = 1 iff the realized move matches the predicted direction:
        is_buy_yes=1  → target=1 iff price@horizon > price_start
        is_buy_yes=0  → target=1 iff price@horizon < price_start
    """
    targets = np.zeros(len(df), dtype=int)
    for i, row in enumerate(df.itertuples(index=False)):
        traj = _read_trajectory(row.signal_id)
        assert horizon_idx >= 1 and horizon_idx <= len(traj), (
            f"horizon_idx={horizon_idx} outside trajectory length {len(traj)}"
        )
        move = traj[horizon_idx - 1] - row.market_price_at_signal
        if row.is_buy_yes == 1:
            targets[i] = int(move > 0)
        else:
            targets[i] = int(move < 0)
    return pd.Series(targets, index=df.index, name=TARGET_COLUMN)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the 1–2 % missing values with column-wise medians (numeric)
    / mode (categorical)."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == "O" or col == "bucket":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    return df


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot the bucket; ensure derived features exist."""
    df = df.copy()
    # Recompute derived features defensively (generator may have NaN'd some)
    df["price_dist_from_0_5"] = (df["market_price_at_signal"] - 0.5).abs()
    df["impact_x_specificity"] = df["impact_strength"] * df["specificity_score"]
    # multi_source_confirmation is already in the CSV; leave as-is

    # One-hot bucket (keep all 5 columns — no drop_first)
    bucket_dummies = pd.get_dummies(df["bucket"], prefix="bucket")
    for b in BUCKETS:
        col = f"bucket_{b}"
        if col not in bucket_dummies.columns:
            bucket_dummies[col] = 0
    df = pd.concat([df.drop(columns=["bucket"]), bucket_dummies], axis=1)
    return df


def _load_processed() -> pd.DataFrame:
    """Read CSV + compute target + cache to parquet for repeated runs.

    Note: the parquet cache is INVALIDATED when generate_dataset.py runs
    (which removes/overwrites raw/signals_export_sample.csv). For correctness
    after a generator run, delete data/processed/dataset.parquet first.
    """
    cache = PROCESSED_DATA_DIR / "dataset.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    csv = RAW_DATA_DIR / "signals_export_sample.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run `python scripts/generate_dataset.py` first."
        )
    df = pd.read_csv(csv, parse_dates=["signal_timestamp"])
    if df["signal_id"].isna().any():
        raise ValueError(
            f"{csv} has {df['signal_id'].isna().sum()} rows with null signal_id — "
            "regenerate with `python scripts/generate_dataset.py`."
        )
    df = _clean(df)
    df[TARGET_COLUMN] = _compute_target(df, HORIZON_REF_IDX)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def _build_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply allowlist + one-hot bucket → X. Build y from TARGET_COLUMN."""
    df_fe = _feature_engineer(df)
    y = df_fe[TARGET_COLUMN].astype(int)

    # Anti-leakage allowlist: 26 named FEATURES minus "bucket" (replaced by 5 one-hot dummies)
    allowed_named = [f for f in FEATURES if f != "bucket"]
    bucket_cols = [f"bucket_{b}" for b in BUCKETS]
    feature_cols = allowed_named + bucket_cols

    # Defensive: none of OUTPUT_COLUMNS sneaks into X
    for out_col in OUTPUT_COLUMNS:
        assert out_col not in feature_cols, f"Output column {out_col} leaked into X"

    X = df_fe[feature_cols].copy()
    return X, y


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Walk-forward split: time-sorted, last 20 % is test.

    Returns (X_train, X_test, y_train, y_test) as numpy arrays, with X
    standardized using a StandardScaler fit on train ONLY (no leakage from test).
    Order: chronological — NO random shuffle.
    """
    df = _load_processed().sort_values("signal_timestamp").reset_index(drop=True)
    X, y = _build_X_y(df)

    n_test = int(len(df) * 0.20)
    split_idx = len(df) - n_test
    X_train_df, X_test_df = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx].to_numpy(), y.iloc[split_idx:].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.to_numpy(dtype=float))
    X_test = scaler.transform(X_test_df.to_numpy(dtype=float))

    return X_train, X_test, y_train, y_test
