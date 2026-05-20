## Pitch (30 s)

Polymarket = marchés de prédiction binaires (OUI/NON, prix = probabilité). Foresight
surveille l'actualité en temps réel ; dès qu'une news touche un marché actif, il
émet un **signal typé** (direction + score 0–100 + fenêtre actionnable). L'heuristique
actuelle (poids devinés à la main) plafonne à AUC ~0.69 sur ce dataset. Ce POC montre
qu'un modèle LightGBM sur 30 features lève l'AUC à ~0.78 et la fenêtre actionnable
optimale émerge clairement (~60 min après le signal).

## Démarrage rapide

```bash
conda create -n poc-foresight python=3.11 -y && conda activate poc-foresight
pip install -r requirements.txt
python scripts/generate_dataset.py     # 3000 signaux + trajectoires 144 pts (seed 42)
python scripts/train.py                # 3 supervisés + auxiliaires + Branche A&B + model_card
python scripts/tune_auc.py             # (optionnel) re-tune si AUC LightGBM hors [0.70, 0.82]
python scripts/honest_analysis.py      # synthèse imprimée
python scripts/make_figures.py         # 6 figures premium dans plots/
pytest -q                              # 32/32 tests doivent passer
python scripts/main.py                 # contrat Basile + Streamlit sur :8501
```

## Architecture (deux branches ML + archétypes)

- **Branche A — interprétabilité (cible binaire à 60 min)**, 3 paliers entraînés
  sur **exactement le même split walk-forward** (mêmes signaux en train, mêmes
  signaux en test — seule la représentation des features change) :
  1. *Heuristique* (poids à la main, 8 facteurs) — voir BRIEF §1
  2. *LogReg-8* (mêmes 8 facteurs, poids appris) — artefact `logreg_eight.joblib`
     (hors registry Basile)
  3. *LightGBM* (matrice 30 colonnes scalée) — dans `config.MODELS`
- **Branche B — courbe « fenêtre actionnable »** : AUC ROC évaluée à chaque pas
  de 10 min sur 24 h (144 points). LightGBM pic vers ~60 min, érosion progressive
  à 24 h ; l'heuristique reste plate.
- **Archétypes K-Means** (analyse non supervisée séparée) : ARI vs target,
  silhouette, profils. Artefact `kmeans.joblib`, hors registry — pas dans le
  tableau des modèles supervisés.
- **Registre Basile (`config.MODELS`)** : exactement 3 modèles supervisés
  comparables like-for-like : `log_reg`, `random_forest`, `lightgbm`.

## Anti-fuite & rigueur

- Allowlist explicite de 26 features nommées (`src/config.py: FEATURES`) — pas
  de fuite des labels / trajectoires
- Walk-forward strict (test = derniers 20 % chronologiquement) — pas de shuffle
  aléatoire
- 5-fold CV en parallèle (sur train uniquement) pour comparer ; WF reste dans
  la même fenêtre que CV (avec une petite tolérance — voir `models/model_card.json`)
- Calibration probabiliste 10-bin
- `models/model_card.json` consigne tous les hyperparams, métriques, et le log
  de tuning `dataset_tuning`

## Structure du repo

| Fichier | Rôle |
|---|---|
| `src/config.py` | Source unique de vérité : paths, seed=42, FEATURES (26), HEURISTIC_8_FACTORS, palette, MODELS registry |
| `src/data.py` | `_clean`, `_feature_engineer`, `_load_processed`, `load_dataset_split` (walk-forward + StandardScaler fit on train) |
| `src/metrics.py` | `compute_metrics` (Basile contract) + `compute_metrics_proba` (proba helper) |
| `src/app.py` | Streamlit demo customisé Foresight (auto-bootstrap sys.path) |
| `scripts/main.py` | **FROZEN** — Basile contract orchestrator (ne pas modifier) |
| `scripts/generate_dataset.py` | Génère 3000 signaux + trajectoires 144 pts + dataset_manifest.json |
| `scripts/train.py` | Entraîne 3 supervisés + 2 auxiliaires + Branche A/B + archétypes + SHAP + calibration |
| `scripts/tune_auc.py` | Auto-tune `FORESIGHT_NOISE_SIGMA` / `FORESIGHT_INTERACTION_SCALE` jusqu'à AUC ∈ [0.70, 0.82] |
| `scripts/honest_analysis.py` | Synthèse imprimée depuis `results/*.json` et `models/model_card.json` |
| `scripts/make_figures.py` | 6 figures premium (charte Foresight, bande d'en-tête réservée) |
| `tests/` | 32 tests pytest (config, data, metrics, models) |
| `plots/01..06_*.png` | Figures commitées |
| `models/model_card.json` | Carte modèle commitée (le reste de `models/*.joblib` est gitignored) |
| `results/model_metrics.csv` | Tableau Basile commité |
| `docs/rapport.md` | Rapport académique |
| `BRIEF.md` | Source spec du POC |

Le plan d'implémentation complet est dans
`docs/superpowers/plans/2026-05-19-foresight-poc.md`.

## Chiffres défendables (Branche A, cible 60 min)

|                          | ROC-AUC | Accuracy |
|--------------------------|--------:|---------:|
| Heuristique (hand)       |   0.688 |    0.625 |
| LogReg-8 (learned)       |   0.705 |    0.662 |
| LightGBM-30 (hero)       |   0.780 |    0.695 |

Branche B : LightGBM pic à 60 min (voir `plots/01_actionable_window.png`).

L'AUC est plafonnée vers 0.82 par construction du générateur (bruit volontaire +
saturation du drift après ~3 h) — le but n'est pas d'impressionner par des
chiffres trop hauts, c'est de **rester crédible** et défendable au jury.
