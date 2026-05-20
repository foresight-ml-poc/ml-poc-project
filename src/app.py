"""Streamlit Foresight POC demo.

Works both via `python scripts/main.py` (Basile contract) and standalone
`streamlit run src/app.py`. The sys.path bootstrap at the top ensures
`from config import ...` resolves either way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    BUCKETS, FEATURES, FORESIGHT_PALETTE, KMEANS_PATH, MODEL_CARD_FILE,
    MODELS, MODEL_METRICS_FILE, PLOTS_DIR, RESULTS_DIR, STEP_MIN, TRAJECTORY_LEN,
)

P = FORESIGHT_PALETTE


def _css():
    st.markdown(
        f"""
        <style>
          .stApp {{ background-color: {P['bg']}; color: {P['ink']}; }}
          .signal-card {{
            background: {P['card']}; border: 1px solid {P['line']};
            border-radius: 12px; padding: 18px 20px;
          }}
          .kpi {{ font-size: 36px; font-weight: 700; color: {P['mint']}; }}
          .kpi-label {{ color: {P['muted']}; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
          .pill-buy {{ background: {P['mint']}; color: {P['bg']}; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }}
          .pill-sell {{ background: {P['loss']}; color: {P['ink']}; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero():
    st.markdown("# Foresight — Real-time news → market signals")
    st.markdown(
        "**Elevator pitch (30 s)** — Polymarket has 100+ active prediction markets "
        "at any moment. Foresight watches the news 24/7 and, the moment information "
        "moves a market, fires a typed signal: direction, conviction score, actionable "
        "window. A hand-tuned heuristic baselined the problem; this POC shows that a "
        "LightGBM model on a 30-column feature matrix lifts AUC from 0.69 → 0.78 and "
        "the winrate (net of spread) into positive territory."
    )


def _basile_table():
    if MODEL_METRICS_FILE.exists():
        st.subheader("Basile contract — model evaluation table")
        st.dataframe(pd.read_csv(MODEL_METRICS_FILE), use_container_width=True)
    else:
        st.info(
            "Run `python scripts/main.py` to generate the Basile evaluation table "
            "(`results/model_metrics.csv`)."
        )


def _branch_a():
    card = json.loads(MODEL_CARD_FILE.read_text())
    a = card["branch_a"]
    st.subheader("Branch A — three rungs of lift @ 60 min")
    rows = []
    for rung in ["heuristic_hand", "logreg_8_learned", "lightgbm_26"]:
        rows.append({"Rung": rung, "ROC-AUC": a[rung]["roc_auc"], "Accuracy": a[rung]["accuracy"]})
    st.dataframe(
        pd.DataFrame(rows).style.format({"ROC-AUC": "{:.3f}", "Accuracy": "{:.3f}"}),
        use_container_width=True,
    )
    st.image(str(PLOTS_DIR / "02_lift_three_rungs.png"))

    branch_a_full = json.loads((RESULTS_DIR / "branch_a_lift.json").read_text())
    hand = branch_a_full["heuristic_hand"]["weights"]
    learned = branch_a_full["logreg_8_learned"]["weights"]
    wdf = pd.DataFrame({
        "Factor": list(hand),
        "Hand weight": [hand[f] for f in hand],
        "Learned weight": [learned[f] for f in hand],
    })
    st.markdown("**Hand vs learned weights (8 factors) — same split, same test set, different feature representation**")
    st.dataframe(
        wdf.style.format({"Hand weight": "{:.4f}", "Learned weight": "{:.4f}"}),
        use_container_width=True,
    )


def _branch_b():
    st.subheader("Branch B — when is the signal actionable?")
    st.image(str(PLOTS_DIR / "01_actionable_window.png"))


def _shap_and_archetypes():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top drivers (SHAP)")
        st.image(str(PLOTS_DIR / "04_shap_beeswarm.png"))
    with col2:
        st.subheader("Signal archetypes")
        st.image(str(PLOTS_DIR / "05_kmeans_archetypes.png"))
        card = json.loads(MODEL_CARD_FILE.read_text())
        arch = card["archetypes"]
        st.caption(
            f"K-Means n=2 · ARI vs target = {arch['ari']:.3f} · "
            f"silhouette = {arch['silhouette']:.3f} · cluster sizes {arch['cluster_sizes']}"
        )


def _backtest():
    st.subheader("Backtest, confusion, calibration")
    st.image(str(PLOTS_DIR / "06_backtest_calibration.png"))


def _signal_demo():
    st.subheader("Live signal demo — score your own signal")
    st.caption(
        "All inputs are the raw feature values; the LightGBM model returns the "
        "calibrated probability that the predicted direction is correct at 60 min. "
        "LightGBM is tree-based — feature scaling doesn't change the prediction."
    )

    lgbm_path = MODELS["lightgbm"]["path"]
    if not lgbm_path.exists():
        st.warning("Run `python scripts/train.py` first to produce the LightGBM model.")
        return
    lgbm = joblib.load(lgbm_path)

    with st.form("signal-form"):
        c1, c2, c3 = st.columns(3)
        impact = c1.slider("impact_strength", 0.0, 1.0, 0.65)
        llm = c1.slider("llm_confidence", 0.0, 1.0, 0.70)
        ambig = c1.slider("ambiguity_score", 0.0, 1.0, 0.25)
        spec = c2.slider("specificity_score", 0.0, 1.0, 0.70)
        cosine = c2.slider("cosine_score", 0.0, 1.0, 0.60)
        novelty = c2.slider("novelty_score", 0.0, 1.0, 0.40)
        sentiment = c3.slider("sentiment_polarity", -1.0, 1.0, 0.10)
        is_buy_yes = c3.selectbox("Direction predicted", ["BUY_YES", "BUY_NO"])
        bucket = c3.selectbox("Bucket", BUCKETS)
        liquidity = st.slider("liquidity_depth (USD)", 100.0, 200_000.0, 10_000.0)
        spread = st.slider("bid_ask_spread", 0.001, 0.1, 0.02)
        price = st.slider("market_price_at_signal", 0.05, 0.95, 0.45)
        ttr = st.slider("time_to_resolution_h", 2.0, 720.0, 120.0)
        submitted = st.form_submit_button("Score signal", type="primary")

    if submitted:
        row = {
            "impact_strength": impact, "llm_confidence": llm, "ambiguity_score": ambig,
            "specificity_score": spec, "cosine_score": cosine, "novelty_score": novelty,
            "sentiment_polarity": sentiment, "articles_count": 5, "unique_sources_count": 4,
            "tier1_count": 1, "tier2_count": 2, "tier3_count": 2, "source_weight": 0.5,
            "freshness_min": 15.0, "market_price_at_signal": price, "bid_ask_spread": spread,
            "liquidity_depth": liquidity, "volatility_pre_24h": 0.05,
            "time_to_resolution_h": ttr,
            "hour_of_day": 12, "day_of_week": 2,
            "is_buy_yes": 1 if is_buy_yes == "BUY_YES" else 0,
            "price_dist_from_0_5": abs(price - 0.5),
            "impact_x_specificity": impact * spec,
            "multi_source_confirmation": min(1.0, (1 + 0.5 * 2) / 5.0),
        }
        for b in BUCKETS:
            row[f"bucket_{b}"] = 1 if b == bucket else 0

        col_order = [f for f in FEATURES if f != "bucket"] + [f"bucket_{b}" for b in BUCKETS]
        x = np.array([[row[c] for c in col_order]], dtype=float)
        proba = float(lgbm.predict_proba(x)[0, 1])

        pill_class = "pill-buy" if is_buy_yes == "BUY_YES" else "pill-sell"
        st.markdown(
            f"""
            <div class='signal-card'>
              <span class='{pill_class}'>{is_buy_yes}</span>
              <span class='kpi-label' style='margin-left:12px'>{bucket}</span>
              <div class='kpi' style='margin-top:8px'>Score {proba*100:.0f}/100</div>
              <div class='kpi-label'>LightGBM predicted P(correct @ 60 min)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_app() -> None:
    """Render the Foresight Streamlit application (Basile contract entry point)."""
    st.set_page_config(page_title="Foresight POC", layout="wide")
    _css()
    _hero()

    st.divider()
    _basile_table()

    st.divider()
    _branch_a()
    st.divider()
    _branch_b()
    st.divider()
    _shap_and_archetypes()
    st.divider()
    _backtest()
    st.divider()
    _signal_demo()


if __name__ == "__main__":
    build_app()
