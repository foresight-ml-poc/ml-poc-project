"""Generate the 6 premium figures from training results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    FORESIGHT_PALETTE, KMEANS_PATH, MODEL_CARD_FILE, MODELS, MODELS_DIR,
    PLOTS_DIR, RESULTS_DIR, STEP_MIN,
)
from data import _build_X_y, _load_processed

P = FORESIGHT_PALETTE
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": P["card"],
    "figure.facecolor": P["bg"],
    "axes.edgecolor": P["line"],
    "axes.labelcolor": P["ink"],
    "xtick.color": P["muted"],
    "ytick.color": P["muted"],
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _setup_canvas(title: str, subtitle: str, figsize=(11, 6.5)):
    """Reserve a header band so titles never overlap the plot area."""
    fig = plt.figure(figsize=figsize, dpi=150)
    header = fig.add_axes([0.06, 0.88, 0.88, 0.10])
    header.axis("off")
    header.text(0, 0.65, title, color=P["ink"], fontsize=20, fontweight="bold", ha="left", va="center")
    header.text(0, 0.10, subtitle, color=P["muted"], fontsize=11, ha="left", va="center")
    ax = fig.add_axes([0.08, 0.10, 0.86, 0.72])
    ax.set_facecolor(P["card"])
    return fig, ax


def _save(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name, dpi=150, facecolor=P["bg"])
    plt.close(fig)


def _split_for_plots():
    """Reproduce the walk-forward split locally — needed for ROC, archetypes, backtest."""
    from sklearn.preprocessing import StandardScaler
    df = _load_processed().sort_values("signal_timestamp").reset_index(drop=True)
    X_df, y = _build_X_y(df)
    n_test = int(len(df) * 0.20)
    split_idx = len(df) - n_test
    X_train_df, X_test_df = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx].to_numpy(), y.iloc[split_idx:].to_numpy()
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
    scaler = StandardScaler().fit(X_train_df.to_numpy(dtype=float))
    X_test = scaler.transform(X_test_df.to_numpy(dtype=float))
    return {
        "X_test": X_test, "y_test": y_test,
        "df_train": df_train, "df_test": df_test,
        "X_test_df": X_test_df,
    }


# --- Figure 1: actionable window (showpiece) ---

def fig_actionable_window():
    curve = json.loads((RESULTS_DIR / "branch_b_curve.json").read_text())
    hm = curve["horizon_minutes"]
    fig, ax = _setup_canvas(
        "When is the signal actionable?",
        "Direction-AUC at every 10-min step over 24 h — heuristic vs LightGBM.",
    )
    ax.plot(hm, curve["heuristic_auc"], color=P["amber"], lw=2.2, label="Heuristic (8 factors)")
    ax.plot(hm, curve["lightgbm_auc"], color=P["mint"], lw=2.8, label="LightGBM (30 features)")
    peak = int(np.argmax(curve["lightgbm_auc"]))
    ax.axvline(hm[peak], color=P["ink"], ls="--", lw=0.8, alpha=0.4)
    ax.annotate(
        f"Peak  AUC = {curve['lightgbm_auc'][peak]:.2f}\nat  {hm[peak]} min",
        xy=(hm[peak], curve["lightgbm_auc"][peak]),
        xytext=(hm[peak] + 120, curve["lightgbm_auc"][peak] - 0.04),
        color=P["ink"], fontsize=11,
        arrowprops=dict(arrowstyle="-", color=P["ink"], lw=0.6),
    )
    ax.set_xlabel("Minutes after signal", color=P["ink"])
    ax.set_ylabel("ROC-AUC", color=P["ink"])
    ax.set_ylim(0.5, 0.85)
    ax.set_xlim(min(hm), max(hm))
    ax.grid(True, color=P["line"], lw=0.4, alpha=0.5)
    ax.legend(facecolor=P["card"], edgecolor=P["line"], labelcolor=P["ink"], loc="lower right")
    _save(fig, "01_actionable_window.png")


# --- Figure 2: lift three rungs ---

def fig_lift_three_rungs():
    a = json.loads((RESULTS_DIR / "branch_a_lift.json").read_text())
    rungs = [
        ("Heuristic\n(hand weights)", a["heuristic_hand"]["roc_auc"], P["amber"]),
        ("LogReg-8\n(learned weights)", a["logreg_8_learned"]["roc_auc"], P["ink"]),
        ("LightGBM-26\n(non-linear)", a["lightgbm_26"]["roc_auc"], P["mint"]),
    ]
    fig, ax = _setup_canvas(
        "Three rungs of lift",
        "Same target, three model families. The non-linear interactions explain the final jump.",
    )
    xs = np.arange(len(rungs))
    vals = [r[1] for r in rungs]
    colors = [r[2] for r in rungs]
    ax.bar(xs, vals, color=colors, width=0.55, edgecolor=P["line"])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.005, f"AUC {v:.3f}", ha="center", color=P["ink"], fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rungs], color=P["ink"])
    ax.set_ylabel("ROC-AUC", color=P["ink"])
    ax.set_ylim(0.55, max(vals) + 0.06)
    ax.grid(True, axis="y", color=P["line"], lw=0.4, alpha=0.5)
    _save(fig, "02_lift_three_rungs.png")


# --- Figure 3: ROC overlay ---

def fig_roc_overlay():
    from sklearn.metrics import roc_curve
    split = _split_for_plots()
    fig, ax = _setup_canvas(
        "ROC overlay — heuristic vs ML family",
        "Bigger area under the curve = stronger ranking. LightGBM dominates.",
    )

    # Heuristic on test set — reuse the train.py HandHeuristic via a local import
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import HandHeuristic
    hh = HandHeuristic()
    proba_h = hh.predict_proba(split["df_test"])[:, 1]
    fpr, tpr, _ = roc_curve(split["y_test"], proba_h)
    ax.plot(fpr, tpr, color=P["amber"], lw=2.0, label="Heuristic")

    for key, color in [
        ("log_reg", P["loss"]),
        ("random_forest", P["muted"]),
        ("lightgbm", P["mint"]),
    ]:
        model = joblib.load(MODELS[key]["path"])
        proba = model.predict_proba(split["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(split["y_test"], proba)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=MODELS[key]["name"])

    ax.plot([0, 1], [0, 1], color=P["line"], ls="--", lw=0.8)
    ax.set_xlabel("False positive rate", color=P["ink"])
    ax.set_ylabel("True positive rate", color=P["ink"])
    ax.grid(True, color=P["line"], lw=0.4, alpha=0.5)
    ax.legend(facecolor=P["card"], edgecolor=P["line"], labelcolor=P["ink"], loc="lower right")
    _save(fig, "03_roc_overlay.png")


# --- Figure 4: SHAP beeswarm ---

def fig_shap_beeswarm():
    import shap
    payload = np.load(RESULTS_DIR / "shap_values.npz", allow_pickle=True)
    values = payload["values"]
    data = payload["data"]
    names = list(payload["feature_names"])

    try:
        fig = plt.figure(figsize=(11, 7.5), dpi=150, facecolor=P["bg"])
        header = fig.add_axes([0.06, 0.92, 0.88, 0.06])
        header.axis("off")
        header.text(0, 0.6, "What drives the LightGBM prediction?",
                    color=P["ink"], fontsize=20, fontweight="bold")
        header.text(0, 0.05, "SHAP beeswarm — top features by impact magnitude.",
                    color=P["muted"], fontsize=11)

        ax = fig.add_axes([0.18, 0.08, 0.78, 0.80])
        ax.set_facecolor(P["card"])
        shap.summary_plot(
            values, features=data, feature_names=names,
            show=False, plot_type="dot", color_bar=True, max_display=12,
        )
        fig.savefig(PLOTS_DIR / "04_shap_beeswarm.png", dpi=150, facecolor=P["bg"])
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] beeswarm failed ({e}), falling back to mean-|SHAP| bar chart")
        plt.close("all")
        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:12]
        fig2, ax2 = _setup_canvas(
            "What drives the LightGBM prediction?",
            "Mean |SHAP| per feature (top 12) — higher = more impact.",
            figsize=(11, 7.5),
        )
        ax2.barh(
            [names[i] for i in order[::-1]],
            mean_abs[order[::-1]],
            color=P["mint"], edgecolor=P["line"],
        )
        ax2.set_xlabel("Mean |SHAP value|", color=P["ink"])
        _save(fig2, "04_shap_beeswarm.png")


# --- Figure 5: K-Means archetypes (PCA 2D) ---

def fig_kmeans_archetypes():
    from sklearn.decomposition import PCA
    split = _split_for_plots()

    km = joblib.load(KMEANS_PATH)  # auxiliary, NOT in config.MODELS
    clusters = km.predict(split["X_test"])
    pca = PCA(n_components=2, random_state=0).fit(split["X_test"])
    coords = pca.transform(split["X_test"])

    fig, ax = _setup_canvas(
        "Signal archetypes (K-Means, n=2)",
        "PCA 2-D projection of test signals. Unsupervised structure ≠ direction prediction.",
    )
    for c, color in zip([0, 1], [P["mint"], P["amber"]]):
        mask = clusters == c
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=14, alpha=0.55,
                   edgecolor="none", label=f"Archetype {c}")
    card = json.loads(MODEL_CARD_FILE.read_text())
    arch = card["archetypes"]
    ax.set_xlabel(f"PC 1   (ARI vs target = {arch['ari']:.3f}; silhouette = {arch['silhouette']:.3f})",
                  color=P["ink"])
    ax.set_ylabel("PC 2", color=P["ink"])
    ax.grid(True, color=P["line"], lw=0.4, alpha=0.5)
    ax.legend(facecolor=P["card"], edgecolor=P["line"], labelcolor=P["ink"], loc="best")
    _save(fig, "05_kmeans_archetypes.png")


# --- Figure 6: backtest + confusion + calibration + winrate ---

def _trajectory_payoff(df_test, y_pred, signal_ids, market_prices, spread_round_trip):
    """Compute realistic Polymarket P&L per signal.

    Trade rule: if y_pred[i]==1, take the position predicted by is_buy_yes[i].
    Position cost = market_price; exit at price[5] (= 60 min); pay round-trip spread.

    Returns: payoff[i] = direction * (price_at_60min - market_price) - spread, if y_pred[i]==1
                       = 0 otherwise.
    """
    from data import _read_trajectory

    payoff = np.zeros(len(df_test), dtype=float)
    for i, (sid, mp, yp, ibuy) in enumerate(zip(
        signal_ids, market_prices, y_pred, df_test["is_buy_yes"].to_numpy()
    )):
        if yp != 1:
            continue
        traj = _read_trajectory(sid)
        direction = 1 if ibuy == 1 else -1
        realized_move = direction * (traj[5] - mp)
        payoff[i] = realized_move - spread_round_trip
    return payoff


def fig_backtest_calibration():
    from sklearn.metrics import confusion_matrix
    from data import _load_processed
    from train import HandHeuristic

    split = _split_for_plots()
    df_test = split["df_test"]
    y_test = split["y_test"]
    market_prices = df_test["market_price_at_signal"].to_numpy()
    signal_ids = df_test["signal_id"].to_numpy()

    # Realistic Polymarket round-trip cost. Brief §5 caps winrate ~60 % net of spread
    # ("plafonné ~60 %, jamais 70 %+ : crédibilité détruite"). 4 % round-trip ≈ 2 %
    # spread chaque côté — réaliste pour un marché mid-tier. Threshold 0.50 = on trade
    # tous les signaux où le modèle penche vers BUY (pas de filtre haute conviction).
    spread_round_trip = 0.04
    threshold = 0.50

    # --- Per-model winrates with realistic payoff ---
    def _model_payoff(model_proba):
        y_pred = (model_proba >= threshold).astype(int)
        payoff = _trajectory_payoff(
            df_test, y_pred, signal_ids, market_prices, spread_round_trip,
        )
        traded = payoff != 0
        winrate = (payoff > 0).sum() / max(traded.sum(), 1)
        return payoff, winrate, y_pred, int(traded.sum())

    # Heuristic
    hh = HandHeuristic()
    proba_h = hh.predict_proba(df_test)[:, 1]
    _, winrate_h, _, n_trades_h = _model_payoff(proba_h)

    # LogReg-8 (auxiliary artifact)
    from config import LOGREG_EIGHT_PATH
    lr8 = joblib.load(LOGREG_EIGHT_PATH)
    norm_test = hh.normalize(df_test)
    proba_lr8 = lr8.predict_proba(norm_test)[:, 1]
    _, winrate_lr8, _, n_trades_lr8 = _model_payoff(proba_lr8)

    # LightGBM (hero)
    lgbm = joblib.load(MODELS["lightgbm"]["path"])
    proba_lgbm = lgbm.predict_proba(split["X_test"])[:, 1]
    payoff_lgbm, winrate_lgbm, y_pred_lgbm, n_trades_lgbm = _model_payoff(proba_lgbm)
    equity = np.cumsum(payoff_lgbm[payoff_lgbm != 0])

    fig = plt.figure(figsize=(13, 9), dpi=150, facecolor=P["bg"])
    header = fig.add_axes([0.06, 0.93, 0.88, 0.05])
    header.axis("off")
    header.text(0, 0.6, "Backtest, confusion, calibration",
                color=P["ink"], fontsize=20, fontweight="bold")
    header.text(0, 0.05,
                f"P&L réaliste : payoff = direction × (prix @ 60 min − prix au signal) "
                f"− {spread_round_trip*100:.1f} % round-trip. Seuil = {threshold}.",
                color=P["muted"], fontsize=11)

    # Equity curve (LightGBM only)
    ax1 = fig.add_axes([0.07, 0.55, 0.55, 0.33])
    ax1.set_facecolor(P["card"])
    ax1.plot(np.arange(len(equity)), equity, color=P["mint"], lw=2.0)
    ax1.axhline(0, color=P["line"], lw=0.6, ls="--")
    ax1.set_title(f"Equity LightGBM ({n_trades_lgbm} trades · winrate {winrate_lgbm*100:.0f} %)",
                  color=P["ink"], loc="left", fontsize=12)
    ax1.set_xlabel("Trades (chronological)", color=P["ink"])
    ax1.set_ylabel("Cumulative P&L (units)", color=P["ink"])
    ax1.grid(True, color=P["line"], lw=0.4, alpha=0.5)

    # Confusion matrix (LightGBM)
    cm = confusion_matrix(y_test, y_pred_lgbm)
    ax2 = fig.add_axes([0.07, 0.07, 0.30, 0.40])
    ax2.set_facecolor(P["card"])
    ax2.imshow(cm, cmap="cividis")
    for (i, j), v in np.ndenumerate(cm):
        ax2.text(j, i, str(v), ha="center", va="center", color=P["ink"], fontweight="bold")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Pred 0", "Pred 1"], color=P["ink"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["True 0", "True 1"], color=P["ink"])
    ax2.set_title("Confusion matrix (LightGBM)", color=P["ink"], loc="left", fontsize=12)

    # Calibration curve
    cal = json.loads((RESULTS_DIR / "calibration.json").read_text())
    ax3 = fig.add_axes([0.45, 0.07, 0.45, 0.40])
    ax3.set_facecolor(P["card"])
    ax3.plot([0, 1], [0, 1], color=P["line"], ls="--", lw=0.8)
    ax3.plot(cal["prob_pred"], cal["prob_true"], color=P["mint"], lw=2.0, marker="o")
    ax3.set_xlabel("Predicted probability", color=P["ink"])
    ax3.set_ylabel("Empirical frequency", color=P["ink"])
    ax3.set_title("Calibration curve (10 bins)", color=P["ink"], loc="left", fontsize=12)
    ax3.grid(True, color=P["line"], lw=0.4, alpha=0.5)

    # Winrate cohorts — ALL THREE computed from data, not hardcoded
    ax4 = fig.add_axes([0.65, 0.55, 0.27, 0.33])
    ax4.set_facecolor(P["card"])
    bars = [
        f"Heuristic\n({n_trades_h} trades)",
        f"LogReg-8\n({n_trades_lr8} trades)",
        f"LightGBM\n({n_trades_lgbm} trades)",
    ]
    vals = [winrate_h, winrate_lr8, winrate_lgbm]
    ax4.bar(bars, vals, color=[P["amber"], P["ink"], P["mint"]], edgecolor=P["line"])
    ax4.axhline(0.50, color=P["loss"], ls=":", lw=1.0)
    for i, v in enumerate(vals):
        ax4.text(i, v + 0.01, f"{v*100:.0f} %", ha="center",
                 color=P["ink"], fontweight="bold")
    ax4.set_ylim(0.40, max(vals) + 0.08)
    ax4.set_title("Winrate net de spread", color=P["ink"], loc="left", fontsize=12)

    fig.savefig(PLOTS_DIR / "06_backtest_calibration.png", dpi=150, facecolor=P["bg"])
    plt.close(fig)


def main() -> None:
    print("Generating 6 premium figures...")
    fig_actionable_window()
    print("  01_actionable_window.png")
    fig_lift_three_rungs()
    print("  02_lift_three_rungs.png")
    fig_roc_overlay()
    print("  03_roc_overlay.png")
    fig_shap_beeswarm()
    print("  04_shap_beeswarm.png")
    fig_kmeans_archetypes()
    print("  05_kmeans_archetypes.png")
    fig_backtest_calibration()
    print("  06_backtest_calibration.png")
    print(f"Done — wrote 6 PNGs to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
