"""Train 3 supervised models (Basile registry) + Branch A & B + archetypes + model card.

Two auxiliary artifacts are saved OUTSIDE config.MODELS:
- LOGREG_EIGHT_PATH: 8-factor LogReg used only by Branch A
- KMEANS_PATH:       2-cluster KMeans used only by the archetype analysis
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from config import (
    BUCKETS, FEATURES, HEURISTIC_8_FACTORS, HEURISTIC_WEIGHTS,
    HORIZON_REF_IDX, KMEANS_PATH, LOGREG_EIGHT_PATH, MODEL_CARD_FILE,
    MODELS, MODELS_DIR, PATHS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR,
    RESULTS_DIR, SEED, STEP_MIN, TARGET_COLUMN, TRAJECTORY_LEN,
)
from data import _build_X_y, _feature_engineer, _load_processed, _read_trajectory  # noqa
from metrics import compute_metrics, compute_metrics_proba


class HandHeuristic:
    """Heuristic baseline that consumes the raw feature dataframe directly."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights if weights is not None else dict(HEURISTIC_WEIGHTS)
        self.factor_order = HEURISTIC_8_FACTORS

    def normalize(self, df: pd.DataFrame) -> np.ndarray:
        out = pd.DataFrame(index=df.index)
        out["freshness_min"] = 1.0 / (1.0 + df["freshness_min"].clip(0) / 30.0)
        out["source_weight"] = df["source_weight"].clip(0, 1)
        out["multi_source_confirmation"] = df["multi_source_confirmation"].clip(0, 1)
        out["impact_strength"] = df["impact_strength"].clip(0, 1)
        out["llm_confidence"] = df["llm_confidence"].clip(0, 1)
        liq = np.log(df["liquidity_depth"].clip(100, 200_000))
        out["liquidity_depth"] = (liq - np.log(100)) / (np.log(200_000) - np.log(100))
        out["bid_ask_spread"] = 1.0 - df["bid_ask_spread"].clip(0, 0.1) / 0.1
        ttr = df["time_to_resolution_h"].clip(2, 720)
        out["time_to_resolution_h"] = np.exp(-((np.log(ttr) - np.log(168)) ** 2) / 2.0)
        return out[self.factor_order].to_numpy(dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        normed = self.normalize(df)
        w = np.array([self.weights[f] for f in self.factor_order])
        score = normed @ w
        return np.column_stack([1.0 - score, score])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)


def _load_split_with_frames():
    df = _load_processed().sort_values("signal_timestamp").reset_index(drop=True)
    X_df, y = _build_X_y(df)

    n_test = int(len(df) * 0.20)
    split_idx = len(df) - n_test

    X_train_df, X_test_df = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx].to_numpy(), y.iloc[split_idx:].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.to_numpy(dtype=float))
    X_test = scaler.transform(X_test_df.to_numpy(dtype=float))

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "df_train": df_train, "df_test": df_test,
        "X_train_df": X_train_df, "X_test_df": X_test_df,
        "scaler": scaler,
    }


def _cv_auc(model_factory, X, y, k=5) -> float:
    """K-fold CV on TRAIN data only (X is already scaled by the train-fit StandardScaler)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    aucs = []
    for train_idx, test_idx in kf.split(X):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs))


def run_branch_a(split: dict) -> dict:
    print("=== Branch A: heuristic → LogReg-8 → LightGBM ===")
    hh = HandHeuristic()
    proba_h = hh.predict_proba(split["df_test"])[:, 1]
    m_h = compute_metrics_proba(split["y_test"], proba_h)

    norm_train = hh.normalize(split["df_train"])
    norm_test = hh.normalize(split["df_test"])
    lr8 = LogisticRegression(max_iter=1000, random_state=SEED)
    lr8.fit(norm_train, split["y_train"])
    joblib.dump(lr8, LOGREG_EIGHT_PATH)
    print(f"  saved auxiliary {LOGREG_EIGHT_PATH.name}")
    proba_lr8 = lr8.predict_proba(norm_test)[:, 1]
    m_lr8 = compute_metrics_proba(split["y_test"], proba_lr8)
    learned_weights = dict(zip(HEURISTIC_8_FACTORS, lr8.coef_[0].tolist()))

    lgbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(split["X_train"], split["y_train"])
    proba_lgbm = lgbm.predict_proba(split["X_test"])[:, 1]
    m_lgbm = compute_metrics_proba(split["y_test"], proba_lgbm)

    return {
        "heuristic_hand": {**m_h, "weights": HEURISTIC_WEIGHTS},
        "logreg_8_learned": {
            **m_lr8,
            "weights": learned_weights,
            "artifact_path": str(LOGREG_EIGHT_PATH.relative_to(MODELS_DIR.parent)),
        },
        "lightgbm_26": m_lgbm,
    }


def run_branch_b(split: dict) -> dict:
    """Branch B stores RAW per-horizon AUC for heuristic AND LightGBM separately.

    NOT a lift. Both lists are 144 raw AUC values, evaluated on direction_correct
    recomputed at each horizon step. LightGBM is refit at each horizon (n_est=200
    for speed inside the loop; the registry LightGBM uses n_est=400).
    """
    print("=== Branch B: 144-pt AUC curve (heuristic vs LightGBM, RAW AUC) ===")
    df_train, df_test = split["df_train"], split["df_test"]
    test_trajs = {row.signal_id: _read_trajectory(row.signal_id) for _, row in df_test.iterrows()}
    train_trajs = {row.signal_id: _read_trajectory(row.signal_id) for _, row in df_train.iterrows()}

    hh = HandHeuristic()
    proba_h = hh.predict_proba(df_test)[:, 1]

    heur_aucs, lgbm_aucs = [], []
    for h_idx in range(1, TRAJECTORY_LEN + 1):
        y_train_h = np.zeros(len(df_train), dtype=int)
        y_test_h = np.zeros(len(df_test), dtype=int)
        for i, row in enumerate(df_train.itertuples(index=False)):
            move = train_trajs[row.signal_id][h_idx - 1] - row.market_price_at_signal
            y_train_h[i] = int((row.is_buy_yes == 1 and move > 0) or (row.is_buy_yes == 0 and move < 0))
        for i, row in enumerate(df_test.itertuples(index=False)):
            move = test_trajs[row.signal_id][h_idx - 1] - row.market_price_at_signal
            y_test_h[i] = int((row.is_buy_yes == 1 and move > 0) or (row.is_buy_yes == 0 and move < 0))

        if len(np.unique(y_test_h)) > 1:
            heur_aucs.append(float(roc_auc_score(y_test_h, proba_h)))
        else:
            heur_aucs.append(0.5)

        lgbm_h = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        lgbm_h.fit(split["X_train"], y_train_h)
        proba_l = lgbm_h.predict_proba(split["X_test"])[:, 1]
        if len(np.unique(y_test_h)) > 1:
            lgbm_aucs.append(float(roc_auc_score(y_test_h, proba_l)))
        else:
            lgbm_aucs.append(0.5)

        if h_idx % 24 == 0:
            print(f"  horizon {h_idx*STEP_MIN:>4} min  heur={heur_aucs[-1]:.3f}  lgbm={lgbm_aucs[-1]:.3f}")

    return {
        "step_min": STEP_MIN,
        "horizon_minutes": [h * STEP_MIN for h in range(1, TRAJECTORY_LEN + 1)],
        "heuristic_auc": heur_aucs,   # RAW heuristic AUC at each horizon
        "lightgbm_auc": lgbm_aucs,    # RAW LightGBM AUC at each horizon
    }


def train_and_save_supervised(split: dict) -> dict:
    """Train the 3 supervised models with the SPEC hyperparameters. No tweaks."""
    print("=== Training 3 supervised models (Basile registry) ===")
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    # SPEC hyperparameters — DO NOT MODIFY
    lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, y_train)
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=10,
        random_state=SEED, n_jobs=-1,
    ).fit(X_train, y_train)
    lgbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=SEED, n_jobs=-1, verbose=-1,
    ).fit(X_train, y_train)

    joblib.dump(lr, MODELS["log_reg"]["path"])
    joblib.dump(rf, MODELS["random_forest"]["path"])
    joblib.dump(lgbm, MODELS["lightgbm"]["path"])

    factories = {
        "log_reg": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            random_state=SEED, n_jobs=-1,
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=SEED, n_jobs=-1, verbose=-1,
        ),
    }
    fitted_models = {"log_reg": lr, "random_forest": rf, "lightgbm": lgbm}

    report = {}
    for key in ["log_reg", "random_forest", "lightgbm"]:
        cv_auc = _cv_auc(factories[key], X_train, y_train)
        proba = fitted_models[key].predict_proba(X_test)[:, 1]
        report[key] = {
            "cv": {"roc_auc": cv_auc},
            "walk_forward": compute_metrics_proba(y_test, proba),
        }
    return report


def run_archetypes(split: dict) -> dict:
    print("=== KMeans archetypes (unsupervised) ===")
    X_train, X_test = split["X_train"], split["X_test"]
    y_test = split["y_test"]

    km = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(X_train)
    joblib.dump(km, KMEANS_PATH)
    print(f"  saved auxiliary {KMEANS_PATH.name}")

    test_clusters = km.predict(X_test)
    silhouette = float(silhouette_score(X_test, test_clusters))
    ari = float(adjusted_rand_score(y_test, test_clusters))
    cluster_sizes = [int(np.sum(test_clusters == c)) for c in range(km.n_clusters)]

    feature_names = list(split["X_test_df"].columns)
    df_test_means = pd.DataFrame(X_test, columns=feature_names)
    df_test_means["_cluster"] = test_clusters
    profiles = {}
    for c in range(km.n_clusters):
        mask = df_test_means["_cluster"] == c
        means = df_test_means[mask].drop(columns=["_cluster"]).mean().sort_values(key=abs, ascending=False).head(8)
        profiles[f"cluster_{c}"] = {k: float(v) for k, v in means.items()}

    return {
        "n_clusters": int(km.n_clusters),
        "silhouette": silhouette,
        "ari": ari,
        "cluster_sizes": cluster_sizes,
        "profiles": profiles,
        "artifact_path": str(KMEANS_PATH.relative_to(MODELS_DIR.parent)),
    }


def compute_shap_and_calibration(split: dict) -> None:
    import shap
    from sklearn.calibration import calibration_curve

    print("=== SHAP + calibration ===")
    lgbm = joblib.load(MODELS["lightgbm"]["path"])
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(split["X_test"])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    feature_names = list(split["X_test_df"].columns)
    np.savez(
        RESULTS_DIR / "shap_values.npz",
        values=shap_values, data=split["X_test"],
        feature_names=np.array(feature_names, dtype=object),
    )

    proba = lgbm.predict_proba(split["X_test"])[:, 1]
    prob_true, prob_pred = calibration_curve(split["y_test"], proba, n_bins=10)
    (RESULTS_DIR / "calibration.json").write_text(json.dumps({
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }))


def main() -> None:
    np.random.seed(SEED)
    print(f"Loading walk-forward split (seed={SEED})...")
    split = _load_split_with_frames()

    branch_a = run_branch_a(split)
    (RESULTS_DIR / "branch_a_lift.json").write_text(json.dumps(branch_a, indent=2))

    report = train_and_save_supervised(split)

    branch_b = run_branch_b(split)
    (RESULTS_DIR / "branch_b_curve.json").write_text(json.dumps(branch_b))

    archetypes = run_archetypes(split)
    compute_shap_and_calibration(split)

    cv_vs_wf = {
        k: {"cv_auc": v["cv"]["roc_auc"], "walk_forward_auc": v["walk_forward"]["roc_auc"]}
        for k, v in report.items()
    }
    (RESULTS_DIR / "cv_vs_walkforward.json").write_text(json.dumps(cv_vs_wf, indent=2))

    card = {
        "seed": SEED,
        "n_signals": len(split["y_train"]) + len(split["y_test"]),
        "n_features": split["X_test"].shape[1],
        "features_allowlist": FEATURES,
        "horizon_ref_min": HORIZON_REF_IDX * STEP_MIN,
        "split": "walk_forward_80_20_time_sorted",
        "models": report,
        "branch_a": {k: {"roc_auc": v["roc_auc"], "accuracy": v["accuracy"]} for k, v in branch_a.items()},
        "branch_b_peak_idx": int(np.argmax(branch_b["lightgbm_auc"])),
        "branch_b_peak_minute": (int(np.argmax(branch_b["lightgbm_auc"])) + 1) * STEP_MIN,
        "archetypes": archetypes,
        "dataset_tuning": {
            "noise_sigma": float(os.environ.get("FORESIGHT_NOISE_SIGMA", "0.85")),
            "interaction_scale": float(os.environ.get("FORESIGHT_INTERACTION_SCALE", "1.0")),
        },
    }
    MODEL_CARD_FILE.write_text(json.dumps(card, indent=2))
    print(f"Model card written to {MODEL_CARD_FILE}")


if __name__ == "__main__":
    main()
