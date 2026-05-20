"""Generate 3000 representative signals + 144-pt price trajectories.

Per brief §3:
- 3000 rows ordered over ~60 days
- 144 points / signal (1 / 10 min over 24 h)
- 26 named features sampled from realistic distributions
- Non-linear latent p with interactions (so ML beats linear heuristic)
- Horizon modulation: edge peaks ~30–120 min, erodes by 24 h (bell curve)
- AUC plafonnée ~0.82, base rate ~52 %
- 1–2 % missing values + a few near-useless features

Tunable parameters (priority order):
1. Env vars FORESIGHT_NOISE_SIGMA / FORESIGHT_INTERACTION_SCALE (used by tune_auc.py)
2. data/raw/dataset_manifest.json if present (deterministic replay of a prior run)
3. Defaults: 0.85 noise sigma, 1.0 interaction scale

After generation, dataset_manifest.json is rewritten with the params actually used —
this is the durable source of truth for "what parameters produced this dataset".
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Bootstrap so we can `from config import ...` (Basile contract pattern)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    BUCKETS, N_SIGNALS, PATHS_DIR, RAW_DATA_DIR, SEED, STEP_MIN,
    TIMESTAMP_SPAN_DAYS, TRAJECTORY_LEN,
)

_MANIFEST_PATH = RAW_DATA_DIR / "dataset_manifest.json"


def _resolve_tuning_params() -> tuple[float, float]:
    """Return (noise_sigma, interaction_scale). Priority: env vars > manifest > defaults.

    Raises ValueError if either resolved param is outside the safe range [0.3, 1.5].
    """
    sigma_default, scale_default = 0.85, 1.0
    if _MANIFEST_PATH.exists():
        try:
            stored = json.loads(_MANIFEST_PATH.read_text())
            sigma_default = float(stored.get("noise_sigma", sigma_default))
            scale_default = float(stored.get("interaction_scale", scale_default))
        except (json.JSONDecodeError, ValueError):
            pass  # corrupted manifest → fall back to defaults
    sigma = float(os.environ.get("FORESIGHT_NOISE_SIGMA", sigma_default))
    scale = float(os.environ.get("FORESIGHT_INTERACTION_SCALE", scale_default))
    # Reject obvious garbage; tune_auc.py's widest attempt is sigma≈1.03, scale=0.85
    if not (0.3 <= sigma <= 1.5):
        raise ValueError(f"FORESIGHT_NOISE_SIGMA={sigma} outside safe range [0.3, 1.5]")
    if not (0.3 <= scale <= 1.5):
        raise ValueError(f"FORESIGHT_INTERACTION_SCALE={scale} outside safe range [0.3, 1.5]")
    return sigma, scale


def _sample_features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # --- Sémantique news ---
    impact_strength = rng.beta(2.0, 3.0, n)                   # [0,1] right-skewed mid
    llm_confidence = rng.beta(4.0, 2.0, n)                    # [0,1] mostly confident
    ambiguity_score = rng.beta(2.5, 4.0, n)                   # [0,1] mostly low
    specificity_score = rng.beta(3.0, 2.5, n)                 # [0,1] mostly high-ish
    cosine_score = rng.beta(3.0, 2.0, n)                      # [0,1]
    novelty_score = rng.beta(2.0, 3.0, n)                     # [0,1] mostly low
    sentiment_polarity = rng.beta(2.0, 2.0, n) * 2 - 1        # [-1,1]

    # --- Sources / crédibilité ---
    articles_count = rng.poisson(5.0, n) + 1                  # >=1
    tier1_count = np.minimum(rng.poisson(1.0, n), articles_count)
    tier2_count = np.minimum(rng.poisson(2.0, n), articles_count - tier1_count)
    tier3_count = np.maximum(articles_count - tier1_count - tier2_count, 0)
    unique_sources_count = np.minimum(
        articles_count, np.maximum(1, rng.poisson(articles_count * 0.6))
    )
    source_weight = np.clip(
        (3.0 * tier1_count + 1.5 * tier2_count + 0.5 * tier3_count) / (articles_count * 3.0),
        0.0, 1.0,
    )
    freshness_min = rng.lognormal(mean=2.5, sigma=0.8, size=n).clip(1, 240)

    # --- Microstructure ---
    market_price_at_signal = rng.beta(2.0, 2.0, n).clip(0.05, 0.95)
    bid_ask_spread = rng.lognormal(-3.5, 0.6, n).clip(0.005, 0.10)
    liquidity_depth = rng.lognormal(7.0, 1.2, n).clip(100, 200_000)
    volatility_pre_24h = rng.lognormal(-3.0, 0.7, n).clip(0.005, 0.20)
    time_to_resolution_h = rng.lognormal(4.0, 1.0, n).clip(2, 24 * 30)

    # --- Contexte ---
    bucket = rng.choice(BUCKETS, n, p=[0.30, 0.20, 0.20, 0.20, 0.10])
    hour_of_day = rng.integers(0, 24, n)
    day_of_week = rng.integers(0, 7, n)
    is_buy_yes = rng.integers(0, 2, n)

    # --- Inject correlations (brief §3.1) ---
    # tier1_count↑ → ambiguity↓
    ambiguity_score = np.clip(ambiguity_score - 0.05 * tier1_count, 0.01, 0.99)
    # liquidity_depth↑ → bid_ask_spread↓
    liq_norm = (np.log(liquidity_depth) - np.log(100)) / (np.log(200_000) - np.log(100))
    bid_ask_spread = np.clip(bid_ask_spread * (1.0 - 0.5 * liq_norm), 0.001, 0.15)

    # --- Dérivées ---
    price_dist_from_0_5 = np.abs(market_price_at_signal - 0.5)
    impact_x_specificity = impact_strength * specificity_score
    multi_source_confirmation = np.clip(
        (tier1_count + 0.5 * tier2_count) / 5.0, 0.0, 1.0,
    )

    df = pd.DataFrame({
        "impact_strength": impact_strength,
        "llm_confidence": llm_confidence,
        "ambiguity_score": ambiguity_score,
        "specificity_score": specificity_score,
        "cosine_score": cosine_score,
        "novelty_score": novelty_score,
        "sentiment_polarity": sentiment_polarity,
        "articles_count": articles_count,
        "unique_sources_count": unique_sources_count,
        "tier1_count": tier1_count,
        "tier2_count": tier2_count,
        "tier3_count": tier3_count,
        "source_weight": source_weight,
        "freshness_min": freshness_min,
        "market_price_at_signal": market_price_at_signal,
        "bid_ask_spread": bid_ask_spread,
        "liquidity_depth": liquidity_depth,
        "volatility_pre_24h": volatility_pre_24h,
        "time_to_resolution_h": time_to_resolution_h,
        "bucket": bucket,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_buy_yes": is_buy_yes,
        "price_dist_from_0_5": price_dist_from_0_5,
        "impact_x_specificity": impact_x_specificity,
        "multi_source_confirmation": multi_source_confirmation,
    })
    return df


def _latent_edge(
    df: pd.DataFrame,
    rng: np.random.Generator,
    noise_sigma: float,
    interaction_scale: float,
) -> np.ndarray:
    """Non-linear latent edge. Returns log-odds (un-sigmoided).

    interaction_scale controls the strength of the non-linear interaction terms
    (impact*specificity, novelty*impact). Used by tune_auc.py to cap the
    LightGBM ceiling: reducing it weakens the gap between linear and tree models
    without inflating noise (cleaner than always bumping sigma).
    """
    # Interaction core (scaled): non-linear gated terms. The `ambig_gate` is a ReLU-like
    # piecewise-linear function (0 above ambiguity≈0.5, ramps up to 0.5 below). Trees
    # capture this with a single split; LogReg can only approximate it via the linear
    # ambiguity coefficient → systematic gap in favor of LightGBM, *even though* LogReg
    # has `impact_x_specificity` as a derived linear feature.
    impact = df["impact_strength"].values
    spec = df["specificity_score"].values
    ambig = df["ambiguity_score"].values
    novelty = df["novelty_score"].values
    llm = df["llm_confidence"].values
    cosine = df["cosine_score"].values
    multi = df["multi_source_confirmation"].values

    ambig_gate = np.clip(0.5 - ambig, 0.0, 1.0)  # 0 when ambig≥0.5, max 0.5 at ambig=0
    interaction = interaction_scale * (
        4.5 * impact * spec * ambig_gate
        + 2.0 * novelty * impact * llm
        + 1.5 * np.where(cosine > 0.55, impact * spec, 0.0)  # threshold gate on cosine
    )
    # Linear core (NOT scaled). Weaker than before so the linear baseline can't outpace
    # the non-linear lift.
    linear = (
        - 0.7 * ambig
        + 0.4 * cosine
        + 0.5 * multi
        + 0.35 * llm
        + 0.3 * impact
    )
    # Microstructure: tight spread + high liquidity favor the move
    spread_norm = np.clip(df["bid_ask_spread"].values / 0.05, 0, 2)
    liq_norm = np.clip(np.log(df["liquidity_depth"].values) / np.log(200_000), 0, 1)
    micro = -0.6 * spread_norm + 0.5 * liq_norm
    # Bucket effect
    bucket_effect = df["bucket"].map({
        "Politics": 0.10, "Geopolitics": 0.05, "Crypto": -0.10,
        "Economy": 0.00, "Sports": -0.05,
    }).fillna(0.0).values
    # Noise (env-tunable); default 0.85 caps AUC ~0.82
    noise = rng.normal(0.0, noise_sigma, len(df))
    # Global shift adjusted to keep base rate ~0.52 given the gated interaction terms
    return interaction + linear + micro + bucket_effect + noise - 0.9


def _horizon_modulation(steps: int) -> np.ndarray:
    """Bell curve: edge weak at t=0, peaks ~30–120 min, erodes by 24 h.

    With STEP_MIN=10 and TRAJECTORY_LEN=144, peak around step 6–12 (60–120 min).
    """
    t = np.arange(1, steps + 1)
    peak = 1.0 * np.exp(-((t - 9) ** 2) / (2 * 14 ** 2))
    erosion = np.clip(1.0 - (t - 9) / 200.0, 0.55, 1.0)
    mod = peak * erosion
    return 0.25 + 0.85 * mod / mod.max()


def _simulate_trajectory(
    edge: float,
    is_buy_yes: int,
    price_start: float,
    volatility: float,
    horizon_mod: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a 144-pt price trajectory. Brownian + drift = horizon_mod[t] * sign * tanh(edge).

    Drift / noise multipliers (0.025 / 0.4) calibrated to land LightGBM walk-forward
    AUC in the brief target window [0.70, 0.82] with default sigma=0.85 + scale=1.0.
    """
    n = len(horizon_mod)
    edge_signed = np.tanh(edge) * 0.025  # max ~2.5 % drift per step at peak
    direction = 1 if is_buy_yes == 1 else -1
    drift = direction * edge_signed * horizon_mod
    noise = rng.normal(0.0, volatility * 0.4, n)
    increments = drift + noise
    prices = np.clip(price_start + np.cumsum(increments), 0.01, 0.99)
    return prices


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator, rate: float = 0.015) -> pd.DataFrame:
    """Inject 1–2 % missing values across numeric columns (brief §3.3)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip = {"signal_id"}
    for col in numeric_cols:
        if col in skip:
            continue
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def main() -> None:
    noise_sigma, interaction_scale = _resolve_tuning_params()
    rng = np.random.default_rng(SEED)
    print(f"Generating {N_SIGNALS} signals with seed={SEED}  "
          f"NOISE_SIGMA={noise_sigma}  INTERACTION_SCALE={interaction_scale}")

    df = _sample_features(rng, N_SIGNALS)

    # Order by signal_timestamp (~60 days span)
    start = pd.Timestamp("2026-03-01")
    minute_offsets = np.sort(rng.uniform(0, TIMESTAMP_SPAN_DAYS * 24 * 60, N_SIGNALS))
    df["signal_timestamp"] = [start + pd.Timedelta(minutes=float(m)) for m in minute_offsets]
    df = df.sort_values("signal_timestamp").reset_index(drop=True)
    df["signal_id"] = [f"sig_{i:05d}" for i in range(N_SIGNALS)]

    edges = _latent_edge(df, rng, noise_sigma, interaction_scale)
    horizon_mod = _horizon_modulation(TRAJECTORY_LEN)

    print(f"Simulating {TRAJECTORY_LEN}-pt trajectories...")
    PATHS_DIR.mkdir(parents=True, exist_ok=True)
    for i, (_, row) in enumerate(df.iterrows()):
        prices = _simulate_trajectory(
            edge=float(edges[i]),
            is_buy_yes=int(row["is_buy_yes"]),
            price_start=float(row["market_price_at_signal"]),
            volatility=float(row["volatility_pre_24h"]),
            horizon_mod=horizon_mod,
            rng=rng,
        )
        traj = {
            "signal_id": row["signal_id"],
            "step_min": STEP_MIN,
            "price_start": float(row["market_price_at_signal"]),
            "price": prices.tolist(),
        }
        (PATHS_DIR / f"{row['signal_id']}.json").write_text(json.dumps(traj))
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{N_SIGNALS} trajectories")

    df["trajectory_path"] = [f"paths/{sid}.json" for sid in df["signal_id"]]

    # Inject realistic dirtiness
    df = _inject_missing(df, rng, rate=0.015)

    out_csv = RAW_DATA_DIR / "signals_export_sample.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows).")
    print(f"Wrote {N_SIGNALS} trajectories to {PATHS_DIR}/")

    # Persist tuning parameters so future runs (without env vars) replay this exact dataset
    manifest = {
        "seed": SEED,
        "n_signals": N_SIGNALS,
        "trajectory_len": TRAJECTORY_LEN,
        "step_min": STEP_MIN,
        "noise_sigma": noise_sigma,
        "interaction_scale": interaction_scale,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {_MANIFEST_PATH} (sigma={noise_sigma}, scale={interaction_scale}).")


if __name__ == "__main__":
    main()
