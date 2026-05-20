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
    MODELS, MODEL_METRICS_FILE, PATHS_DIR, PLOTS_DIR, RAW_DATA_DIR,
    RESULTS_DIR, STEP_MIN, TARGET_COLUMN, TRAJECTORY_LEN,
)

P = FORESIGHT_PALETTE


# ---------------------------------------------------------------------------
# Cache loaders (Streamlit reruns this whole file on every interaction —
# wrap expensive reads in @st.cache_data so the app stays snappy)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_card():
    return json.loads(MODEL_CARD_FILE.read_text())


@st.cache_data
def _load_branch_a():
    return json.loads((RESULTS_DIR / "branch_a_lift.json").read_text())


@st.cache_data
def _load_signals_csv():
    csv = RAW_DATA_DIR / "signals_export_sample.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv, parse_dates=["signal_timestamp"])


@st.cache_data
def _load_trajectory(signal_id: str):
    path = PATHS_DIR / f"{signal_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def _load_processed_target_rate():
    """Read the processed parquet (cached by data.py) and return the global target rate."""
    from data import _load_processed
    try:
        df = _load_processed()
        return float(df[TARGET_COLUMN].mean()), len(df)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def _css():
    st.markdown(
        f"""
        <style>
          .stApp {{ background-color: {P['bg']}; color: {P['ink']}; }}
          h1, h2, h3, h4 {{ color: {P['ink']}; }}
          .lead {{ color: {P['ink']}; font-size: 17px; line-height: 1.55; max-width: 76ch; }}
          .muted {{ color: {P['muted']}; }}
          .stage {{ color: {P['mint']}; font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }}
          .signal-card {{
            background: {P['card']}; border: 1px solid {P['line']};
            border-radius: 12px; padding: 18px 20px;
          }}
          .kpi {{ font-size: 36px; font-weight: 700; color: {P['mint']}; }}
          .kpi-label {{ color: {P['muted']}; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
          .pill-buy {{ background: {P['mint']}; color: {P['bg']}; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }}
          .pill-sell {{ background: {P['loss']}; color: {P['ink']}; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }}
          .pill-neutral {{ background: {P['line']}; color: {P['ink']}; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stage(num: int, total: int, label: str):
    st.markdown(
        f"<div class='stage'>Étape {num}/{total} · {label}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 1: Le problème
# ---------------------------------------------------------------------------

def _section_product():
    """Section 1 — hook the audience: explain Foresight in 30 seconds before
    diving into anything technical."""
    st.markdown("# Foresight")
    st.markdown(
        "<div class='muted' style='margin-top:-12px;font-size:14px;'>"
        "POC ML · signaux Polymarket · démonstration scolaire</div>",
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;")
    _stage(1, 8, "Le produit")
    st.markdown("## En 30 secondes")
    col_text, col_card = st.columns([3, 2])
    with col_text:
        st.markdown(
            "<div class='lead'>"
            "<strong>Polymarket</strong> est une plateforme où des milliers de "
            "gens parient en temps réel sur l'actualité — élections, prix du "
            "Bitcoin, sortie d'un produit. Chaque marché est binaire : on achète "
            "« OUI » ou « NON », le prix se balade entre 0 et 1 et représente "
            "la probabilité implicite vue par la foule. À la résolution, le bon "
            "côté vaut 1 $, l'autre 0 $.<br><br>"
            "<strong>Foresight</strong> surveille la presse 24/7. Dès qu'une "
            "news touche un de ces marchés, le pipeline fabrique en quelques "
            "secondes un <strong>signal typé</strong> : direction (acheter OUI "
            "ou acheter NON), score de conviction 0–100, fenêtre d'action. Il "
            "en sort plusieurs par heure aux pics d'actualité.<br><br>"
            "Aujourd'hui ces signaux sont scorés par une formule à 8 facteurs "
            "aux poids devinés à la main. Ça marche, mais ça plafonne. "
            "<strong>Ce POC montre comment le ML peut faire mieux</strong> — et "
            "surtout, nous dire <em>quand</em> agir."
            "</div>",
            unsafe_allow_html=True,
        )
    with col_card:
        st.markdown(
            f"""
            <div class='signal-card' style='border-color:{P['mint']}'>
              <div style='display:flex;gap:10px;align-items:center'>
                <span class='pill-buy'>BUY_YES</span>
                <span class='kpi-label'>Politics · NYC mayoral race</span>
              </div>
              <div class='kpi' style='margin-top:10px'>84 / 100</div>
              <div class='kpi-label'>conviction LightGBM @ 60 min</div>
              <div class='muted' style='margin-top:14px;font-size:13px'>
                impact 0,72 · spécificité 0,80 · ambiguïté 0,18<br>
                liq 18 k$ · spread 1,8 %
              </div>
            </div>
            <div class='muted' style='font-size:12px;text-align:center;margin-top:8px'>
              Exemple de carte signal Foresight
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section 2: Le problème
# ---------------------------------------------------------------------------

def _section_problem():
    _stage(2, 8, "Le problème")
    st.markdown("## L'heuristique plafonne. On laisse des trades sur la table.")
    st.markdown(
        "<div class='lead'>"
        "La formule actuelle <strong>pèse 8 facteurs linéairement</strong>, avec "
        "des poids choisis à l'œil par les ingénieurs Foresight. La voici telle "
        "qu'elle tourne en production :"
        "</div>",
        unsafe_allow_html=True,
    )
    st.code(
        """signal_score = 100 × (
    0.75 × ( 0.15·freshness + 0.10·source_weight + 0.15·confirmation
           + 0.60·( 0.65·impact_strength + 0.35·llm_confidence ) )
  + 0.25 × ( 0.40·liquidity + 0.35·spread + 0.25·time_to_resolution )
)""",
        language="text",
    )
    st.markdown(
        "<div class='lead'>"
        "En développant les parenthèses, on obtient <strong>8 poids effectifs</strong> "
        "sur les 8 facteurs (somme = 1) :"
        "</div>",
        unsafe_allow_html=True,
    )
    from config import HEURISTIC_WEIGHTS
    weights_df = pd.DataFrame({
        "Facteur": list(HEURISTIC_WEIGHTS.keys()),
        "Poids effectif": list(HEURISTIC_WEIGHTS.values()),
    })
    st.dataframe(
        weights_df.style.format({"Poids effectif": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown("&nbsp;")
    st.markdown(
        "<div class='lead'>"
        "<strong>Elle plafonne à AUC ~0,69 sur le test set</strong> — autant dire "
        "qu'elle rate les interactions entre variables (par ex. l'impact ne compte "
        "vraiment que si la news est spécifique <em>et</em> peu ambiguë). Et elle "
        "ne dit rien sur le bon moment pour agir : elle donne le même score quelle "
        "que soit la fenêtre temporelle. <strong>Deux limites, deux questions "
        "auxquelles ce POC va répondre.</strong>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='signal-card'>"
            f"<div class='kpi-label'>AUC heuristique actuelle</div>"
            f"<div class='kpi' style='color:{P['amber']}'>0,69</div>"
            f"<div class='muted' style='font-size:13px;margin-top:4px;'>poids devinés, 8 facteurs linéaires</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='signal-card'>"
            f"<div class='kpi-label'>Winrate net (estimé)</div>"
            f"<div class='kpi' style='color:{P['amber']}'>~55 %</div>"
            f"<div class='muted' style='font-size:13px;margin-top:4px;'>après spread, juste au-dessus du break-even</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='signal-card'>"
            f"<div class='kpi-label'>Question</div>"
            f"<div style='font-size:18px;color:{P['ink']};margin-top:6px;line-height:1.4'>"
            f"Le ML peut-il faire <strong>mieux</strong> que l'heuristique &mdash; "
            f"et nous dire <strong>quand</strong> agir ?"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section 3: Les données
# ---------------------------------------------------------------------------

def _section_data():
    _stage(3, 8, "Les données")
    st.markdown("## Ce qu'on analyse — 3 000 signaux, 60 jours, 24 h de prix par signal")
    st.markdown(
        "<div class='lead'>"
        "Dataset représentatif simulé (seed 42, généré par "
        "<code>scripts/generate_dataset.py</code>) : 3 000 signaux "
        "ordonnés sur ~60 jours, <strong>26 features nommées par signal</strong> "
        "(sémantique news, sources, microstructure, contexte) et une "
        "<strong>trajectoire de prix sur 24 h</strong> par signal — 144 points, "
        "un toutes les 10 minutes. C'est cette trajectoire qui sert ensuite à "
        "calculer la cible binaire à 60 min."
        "</div>",
        unsafe_allow_html=True,
    )

    df = _load_signals_csv()
    if df is None:
        st.warning(
            "Lance `python scripts/generate_dataset.py` pour générer les données "
            "(elles sont gitignored par défaut)."
        )
        return

    target_rate, n_total = _load_processed_target_rate()

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Signaux", f"{n_total or len(df):,}")
    k2.metric("Features (allowlist)", "26 nommées → 30 cols")
    k3.metric("Trajectoires", f"{TRAJECTORY_LEN} pts · 1 / {STEP_MIN} min")
    if target_rate is not None:
        k4.metric("Base rate cible @60 min", f"{target_rate*100:.1f} %")

    st.markdown("&nbsp;")

    # Sample of 8 signals
    st.markdown("**Un échantillon — 8 signaux pris au hasard** (8 colonnes sur 30) :")
    sample_cols = [
        "signal_id", "signal_timestamp", "bucket", "is_buy_yes",
        "impact_strength", "llm_confidence", "market_price_at_signal",
        "freshness_min",
    ]
    avail = [c for c in sample_cols if c in df.columns]
    sample = df.sample(8, random_state=7)[avail].reset_index(drop=True).copy()
    if "is_buy_yes" in sample.columns:
        sample["is_buy_yes"] = sample["is_buy_yes"].map({1: "BUY_YES", 0: "BUY_NO"})
    fmt = {}
    for col in ["impact_strength", "llm_confidence", "market_price_at_signal"]:
        if col in sample.columns:
            fmt[col] = "{:.3f}"
    if "freshness_min" in sample.columns:
        fmt["freshness_min"] = "{:.1f}"
    st.dataframe(sample.style.format(fmt), use_container_width=True)

    # One trajectory plot
    st.markdown(
        "**Une trajectoire** &mdash; prix d'un marché toutes les 10 min sur 24 h, "
        "à partir du moment où le signal est tombé :"
    )
    sample_id = df.iloc[500]["signal_id"]
    is_buy_yes = int(df.iloc[500]["is_buy_yes"])
    direction_label = "BUY_YES" if is_buy_yes == 1 else "BUY_NO"
    traj = _load_trajectory(sample_id)
    if traj is not None:
        try:
            import plotly.graph_objects as go
            minutes = [i * STEP_MIN for i in range(1, len(traj["price"]) + 1)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=minutes, y=traj["price"], mode="lines",
                line=dict(color=P["mint"], width=2.2),
                name=f"Prix {sample_id}",
            ))
            fig.add_hline(
                y=traj["price_start"], line_dash="dash", line_color=P["muted"],
                annotation_text=f"prix au signal · {traj['price_start']:.3f}",
                annotation_font_color=P["muted"],
                annotation_position="top right",
            )
            fig.add_vline(
                x=60, line_dash="dot", line_color=P["amber"],
                annotation_text="60 min · cible Branche A",
                annotation_font_color=P["amber"],
                annotation_position="top",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=P["bg"], plot_bgcolor=P["card"],
                xaxis_title="Minutes après le signal",
                yaxis_title="Prix marché",
                height=340,
                margin=dict(l=40, r=30, t=30, b=40),
                showlegend=False,
                title=dict(
                    text=f"{sample_id} · {direction_label} · bucket {df.iloc[500]['bucket']}",
                    font=dict(size=13, color=P["muted"]),
                    x=0.0,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(pd.DataFrame({"price": traj["price"]}))
        st.caption(
            "Cette trajectoire est l'objet canonique : tout (cible à 60 min, "
            "courbe d'AUC sur 24 h) en est dérivé. On ne stocke jamais de snapshots — "
            "une seule granularité, zéro redondance."
        )
    else:
        st.info(f"Trajectoire de {sample_id} introuvable (run `generate_dataset.py` first).")


# ---------------------------------------------------------------------------
# Section 4: Branche A — peut-on faire mieux ?
# ---------------------------------------------------------------------------

def _section_branch_a():
    _stage(4, 8, "Première question")
    st.markdown("## Peut-on faire mieux que l'heuristique sur la même donnée ?")
    st.markdown(
        "<div class='lead'>"
        "Une heuristique et deux modèles ML, <strong>évalués sur exactement le "
        "même test set</strong> (mêmes 600 signaux, même cible à 60 min). "
        "LogReg-8 et LightGBM sont entraînés sur les 2400 signaux du train ; "
        "l'heuristique, elle, a des poids fixés à la main — pas de phase "
        "d'apprentissage, pas besoin du train set. Seule la "
        "<strong>représentation</strong> des features change : l'heuristique et "
        "LogReg-8 voient les 8 facteurs normalisés du brief §1 ; LightGBM voit "
        "la matrice complète à 30 colonnes et peut modéliser les interactions. "
        "Le lift qu'on voit n'est donc pas un artefact de splits différents."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;")

    card = _load_card()
    a = card["branch_a"]
    rows = [
        {"Palier": "Heuristique (poids main, 8 facteurs)", "ROC-AUC": a["heuristic_hand"]["roc_auc"], "Accuracy": a["heuristic_hand"]["accuracy"]},
        {"Palier": "LogReg-8 (mêmes 8 facteurs, poids appris)", "ROC-AUC": a["logreg_8_learned"]["roc_auc"], "Accuracy": a["logreg_8_learned"]["accuracy"]},
        {"Palier": "LightGBM (30 features, non-linéaire)", "ROC-AUC": a["lightgbm_26"]["roc_auc"], "Accuracy": a["lightgbm_26"]["accuracy"]},
    ]
    st.dataframe(
        pd.DataFrame(rows).style.format({"ROC-AUC": "{:.3f}", "Accuracy": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    lift = a["lightgbm_26"]["roc_auc"] - a["heuristic_hand"]["roc_auc"]
    st.markdown(
        f"<div class='lead'>"
        f"<strong>Lift global : +{lift*100:.1f} points d'AUC</strong> entre "
        f"l'heuristique de base et le modèle non-linéaire. Le repesage seul "
        f"(LogReg-8) couvre une partie du chemin ; le reste est purement "
        f"non-linéaire — c'est ce que les interactions entre features apportent."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.image(str(PLOTS_DIR / "02_lift_three_rungs.png"))

    branch_a_full = _load_branch_a()
    hand = branch_a_full["heuristic_hand"]["weights"]
    learned = branch_a_full["logreg_8_learned"]["weights"]
    wdf = pd.DataFrame({
        "Facteur": list(hand),
        "Poids hand (heuristique)": [hand[f] for f in hand],
        "Poids appris (LogReg-8)": [learned[f] for f in hand],
    })
    st.markdown(
        "**Hand vs appris** &mdash; les facteurs qui drivent <em>vraiment</em> "
        "la prédiction (échelle des poids appris non normalisée) :"
    )
    st.dataframe(
        wdf.style.format({"Poids hand (heuristique)": "{:.4f}", "Poids appris (LogReg-8)": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Lecture rapide : `impact_strength` et `bid_ask_spread` sont sous-pesés "
        "dans l'heuristique. LogReg les remonte au sommet. C'est un signal "
        "produit, pas juste un chiffre."
    )


# ---------------------------------------------------------------------------
# Section 5: Branche B — quand faut-il agir ?
# ---------------------------------------------------------------------------

def _section_branch_b():
    _stage(5, 8, "Deuxième question")
    st.markdown("## Quand est-ce actionable ?")
    card = _load_card()
    peak_min = card.get("branch_b_peak_minute", 60)
    st.markdown(
        f"<div class='lead'>"
        f"L'AUC à 60 min, c'est utile. Mais à <em>tous</em> les horizons ? "
        f"On évalue l'AUC à chaque pas de 10 minutes sur 24 h — 144 points. "
        f"La courbe est sans équivoque : <strong>la fenêtre actionnable se "
        f"referme vite</strong>. Pic à {peak_min} minutes après le signal, "
        f"puis érosion progressive parce que le marché digère l'info."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.image(str(PLOTS_DIR / "01_actionable_window.png"))
    st.caption(
        "L'heuristique reste plate (~0,62–0,70) à tous les horizons : elle "
        "score un signal indépendamment du timing. LightGBM, lui, identifie "
        "la fenêtre courte où le mouvement est encore prédictible. C'est "
        "l'insight produit principal du POC."
    )


# ---------------------------------------------------------------------------
# Section 6: Drivers (SHAP) + Archétypes (K-Means)
# ---------------------------------------------------------------------------

def _section_drivers_and_archetypes():
    _stage(6, 8, "La boîte noire ouverte")
    st.markdown("## Pourquoi le modèle prédit ce qu'il prédit")
    st.markdown(
        "<div class='lead'>"
        "Deux angles. À gauche, SHAP : pour chaque signal, quels features ont "
        "poussé la prédiction et dans quel sens. À droite, K-Means non "
        "supervisé : si on oublie la cible et qu'on demande au modèle "
        "« combien de familles vois-tu ? », il en trouve deux — mais ces "
        "familles ne correspondent <em>pas</em> à la direction (ARI ≈ 0). "
        "Elles segmentent autre chose, probablement la microstructure."
        "</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top drivers (SHAP, LightGBM)")
        st.image(str(PLOTS_DIR / "04_shap_beeswarm.png"))
        st.caption(
            "Chaque point = un signal du test set. Couleur = valeur du "
            "feature ; position horizontale = contribution à la prédiction."
        )
    with col2:
        st.subheader("Archétypes K-Means (non supervisé)")
        st.image(str(PLOTS_DIR / "05_kmeans_archetypes.png"))
        arch = _load_card()["archetypes"]
        st.caption(
            f"ARI vs target = {arch['ari']:.3f} (≈ 0 : les clusters ne sont pas "
            f"la direction) · silhouette = {arch['silhouette']:.3f} · "
            f"tailles {arch['cluster_sizes']}. Honnêteté méthodo : on rapporte "
            f"K-Means avec les métriques unsupervised, pas avec une accuracy "
            f"trompeuse."
        )


# ---------------------------------------------------------------------------
# Section 7: Mise à l'épreuve
# ---------------------------------------------------------------------------

def _section_proof():
    _stage(7, 8, "Mise à l'épreuve")
    st.markdown("## L'AUC c'est joli. Le P&L c'est la vérité.")
    st.markdown(
        "<div class='lead'>"
        "Backtest réaliste sur le test set, avec une comparaison <strong>juste à "
        "volume égal</strong> : chaque approche trade ses <strong>300 signaux les plus "
        "convaincants</strong> (= moitié du test). Pour chaque trade, on simule "
        "l'entrée au prix marché et la sortie 60 minutes plus tard sur la trajectoire "
        "enregistrée. <strong>Payoff = direction × (prix @ 60 min − prix d'entrée) "
        "− 4 % de spread round-trip</strong> (coût réaliste pour un marché Polymarket "
        "mid-tier). Un trade « directionnellement correct mais mouvement trop petit » "
        "devient perdant — c'est exactement ce que le brief définit par « net de "
        "spread »."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;")
    st.markdown(
        "<div class='lead'>"
        "<strong>Pourquoi top-K et pas un seuil fixe ?</strong> Parce que les modèles "
        "ont des distributions de score différentes : à seuil 0,50, LogReg-8 inclut "
        "plus de signaux marginaux que l'heuristique, ce qui tire son winrate vers "
        "le bas par effet de volume — pas de ranking. Le top-K élimine ce biais "
        "(même volume partout) et fait apparaître la vraie hiérarchie : LightGBM "
        "range mieux ses meilleurs paris."
        "</div>",
        unsafe_allow_html=True,
    )
    st.image(str(PLOTS_DIR / "06_backtest_calibration.png"))
    st.caption(
        "Les trois winrates en bas à droite sont **calculés depuis la donnée** "
        "à volume égal (top-300 par modèle). Equity curve en haut à gauche = P&L "
        "cumulé LightGBM trade par trade, chronologique. Calibration en bas à "
        "droite : plus la courbe colle à la diagonale, plus les probabilités du "
        "modèle sont fiables."
    )


# ---------------------------------------------------------------------------
# Section 8: Démo live
# ---------------------------------------------------------------------------

def _section_demo():
    _stage(8, 8, "Essayez vous-même")
    st.markdown("## Scorez votre propre signal")
    st.markdown(
        "<div class='lead'>"
        "Donnez les caractéristiques d'un signal imaginaire — LightGBM "
        "renvoie sa probabilité que la direction prédite soit correcte à "
        "60 min. (LightGBM est tree-based : la mise à l'échelle des "
        "features ne change pas la prédiction, donc on peut envoyer les "
        "valeurs brutes.)"
        "</div>",
        unsafe_allow_html=True,
    )
    lgbm_path = MODELS["lightgbm"]["path"]
    if not lgbm_path.exists():
        st.warning("Lance `python scripts/train.py` d'abord pour produire le modèle LightGBM.")
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
        is_buy_yes = c3.selectbox("Direction prédite", ["BUY_YES", "BUY_NO"])
        bucket = c3.selectbox("Bucket marché", BUCKETS)
        liquidity = st.slider("liquidity_depth (USD)", 100.0, 200_000.0, 10_000.0)
        spread = st.slider("bid_ask_spread", 0.001, 0.1, 0.02)
        price = st.slider("market_price_at_signal", 0.05, 0.95, 0.45)
        ttr = st.slider("time_to_resolution_h", 2.0, 720.0, 120.0)
        submitted = st.form_submit_button("Scorer le signal", type="primary")

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
        verdict_color = P["mint"] if proba >= 0.6 else (P["amber"] if proba >= 0.5 else P["loss"])
        verdict_text = "conviction forte" if proba >= 0.6 else ("zone grise" if proba >= 0.5 else "ne pas trader")
        st.markdown(
            f"""
            <div class='signal-card' style='margin-top:18px;border-color:{verdict_color}'>
              <span class='{pill_class}'>{is_buy_yes}</span>
              <span class='kpi-label' style='margin-left:12px'>{bucket}</span>
              <div class='kpi' style='margin-top:8px;color:{verdict_color}'>
                Score {proba*100:.0f} / 100
              </div>
              <div class='kpi-label'>LightGBM · P(direction correcte @ 60 min) · {verdict_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section 9: La solution / synthèse
# ---------------------------------------------------------------------------

def _section_solution():
    st.markdown(" ")
    st.markdown(
        f"<div class='stage'>Conclusion · ce qu'on retient</div>",
        unsafe_allow_html=True,
    )
    st.markdown("## La solution : un modèle qui dit aussi *quand* agir")

    card = _load_card()
    a = card["branch_a"]
    heur_auc = a["heuristic_hand"]["roc_auc"]
    lgbm_auc = a["lightgbm_26"]["roc_auc"]
    lift_pts = (lgbm_auc - heur_auc) * 100
    peak_min = card.get("branch_b_peak_minute", 60)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='signal-card' style='border-color:{P['mint']}'>"
            f"<div class='kpi-label'>Lift d'AUC vs heuristique</div>"
            f"<div class='kpi'>+{lift_pts:.1f} pts</div>"
            f"<div class='muted' style='font-size:13px;margin-top:6px'>"
            f"{heur_auc:.2f} &rarr; {lgbm_auc:.2f} sur la même donnée"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='signal-card' style='border-color:{P['mint']}'>"
            f"<div class='kpi-label'>Fenêtre actionnable</div>"
            f"<div class='kpi'>{peak_min} min</div>"
            f"<div class='muted' style='font-size:13px;margin-top:6px'>"
            f"après le signal, avant que le marché digère"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='signal-card' style='border-color:{P['mint']}'>"
            f"<div class='kpi-label'>Méthodo</div>"
            f"<div class='kpi'>0 fuite</div>"
            f"<div class='muted' style='font-size:13px;margin-top:6px'>"
            f"allowlist 26 features · walk-forward strict · CV ≈ WF"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;")
    st.markdown(
        "<div class='lead'>"
        "<strong>Le pitch en une phrase</strong> &mdash; on ne remplace pas "
        "l'heuristique pour la beauté du chiffre. On la remplace parce que "
        "le modèle non-linéaire dit aussi <em>quand</em> agir, et c'est "
        "<em>ça</em> qui fait basculer le winrate net de spread du break-even "
        "vers une espérance positive."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("&nbsp;")
    st.markdown("**Prochaines étapes** (post-POC) :")
    st.markdown(
        "- Backfill du dataset depuis l'API Polymarket CLOB déjà branchée en prod\n"
        "- Threshold de décision tuné par bucket de marché (Politics ≠ Crypto)\n"
        "- Ensemble voting LightGBM + RF + LogReg pour la robustesse\n"
        "- Drift detection sur les distributions de features (la presse change)\n"
        "- A/B test live contre l'heuristique : le seul jugement qui compte"
    )


# ---------------------------------------------------------------------------
# Annex: Basile contract evaluation table
# ---------------------------------------------------------------------------

def _section_basile_annex():
    with st.expander("Annexe — tableau d'évaluation Basile (contrat scolaire)"):
        if MODEL_METRICS_FILE.exists():
            st.dataframe(pd.read_csv(MODEL_METRICS_FILE), use_container_width=True)
            st.caption(
                "Tableau produit par `scripts/main.py` (orchestrateur Basile). "
                "L'AUC ici est calculée sur les prédictions *dures* (`.predict()`) "
                "et est donc dégénérée — la vraie AUC probabiliste vit dans "
                "`models/model_card.json` (et le ROC overlay de la section "
                "« Boîte noire »)."
            )
        else:
            st.info(
                "Lance `python scripts/main.py` pour générer le tableau "
                "(`results/model_metrics.csv`)."
            )


# ---------------------------------------------------------------------------
# Entry point (Basile contract)
# ---------------------------------------------------------------------------

def build_app() -> None:
    """Render the Foresight Streamlit application (Basile contract entry point)."""
    st.set_page_config(page_title="Foresight POC", layout="wide")
    _css()

    _section_product()
    st.divider()
    _section_problem()
    st.divider()
    _section_data()
    st.divider()
    _section_branch_a()
    st.divider()
    _section_branch_b()
    st.divider()
    _section_drivers_and_archetypes()
    st.divider()
    _section_proof()
    st.divider()
    _section_demo()
    st.divider()
    _section_solution()
    st.divider()
    _section_basile_annex()


if __name__ == "__main__":
    build_app()
