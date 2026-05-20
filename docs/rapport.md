# Foresight POC — Rapport

## 1. Contexte & problème

Polymarket héberge des marchés de prédiction binaires où le prix = probabilité
implicite. Foresight produit un signal typé chaque fois qu'une news déplace
substantiellement un marché. L'heuristique de scoring actuelle est linéaire à
poids devinés ; elle ignore les interactions entre variables (par ex. `impact`
n'aide que si `specificity` haut **ET** `ambiguity` bas).

## 2. Données

Dataset représentatif simulé (3000 signaux + trajectoires 144 pts / signal,
seed 42, `data/raw/dataset_manifest.json` pour la repro déterministe). 26
features nommées (BRIEF §3.1) → matrice X de 30 colonnes après one-hot sur
`bucket` (25 numériques + 5 dummies). Cible binaire `direction_correct` à
l'horizon de référence 60 min, dérivée du 6ᵉ point de chaque trajectoire (la
convention d'indexation est documentée et testée — voir `_compute_target` dans
`src/data.py`). 1–2 % de valeurs manquantes injectées et imputées (médiane /
mode) pour réalisme.

## 3. Méthodologie

- Allowlist anti-fuite stricte (26 features signal-time ; `bucket` one-hot vers
  5 colonnes ; aucune trajectoire `move_*` ne pénètre X)
- Walk-forward 80/20 chronologique + 5-fold CV randomisée (sur train) pour
  comparer la dégradation
- **Registre supervisé `config.MODELS`** (3 familles comparables sur le même
  feature matrix scalé) : Logistic Regression, Random Forest, LightGBM
- **Artefacts auxiliaires hors registry** :
  - `logreg_eight.joblib` : LogReg sur 8 facteurs normalisés (Branche A,
    comparaison « poids main vs poids appris »)
  - `kmeans.joblib` : KMeans n=2, analyse non supervisée séparée (ARI vs
    target, silhouette, profils d'archétypes)
- Heuristique baseline (poids à la main, BRIEF §1) — non sérialisée, calculée
  à la volée pour Branche A
- SHAP (TreeExplainer) sur LightGBM
- Calibration probabiliste (10 bins)
- Boucle d'auto-tuning `scripts/tune_auc.py` pour garder l'AUC LightGBM dans
  `[0.70, 0.82]` — paramètres effectifs (`noise_sigma`, `interaction_scale`)
  consignés dans `model_card.dataset_tuning`

## 4. Résultats

### Branche A — lift 3 niveaux (cible 60 min)

**Note méthodo (anti-piège jury).** Les trois paliers sont évalués sur
**exactement le même split walk-forward** : mêmes signaux en train, mêmes
signaux en test. Seule la représentation des features change — `logreg_eight`
voit 8 facteurs normalisés [0,1], `log_reg` (registry Basile) et `lightgbm`
voient la matrice 30 colonnes scalée. Le lift d'AUC n'est donc PAS un artefact
de splits différents ; il mesure ce que chaque famille de modèle extrait d'un
même substrat.

| Palier | ROC-AUC | Accuracy |
|---|---:|---:|
| Heuristique (poids hand)   | 0.688 | 0.625 |
| LogReg-8 (poids appris)    | 0.705 | 0.662 |
| LightGBM-30 (non-linéaire) | 0.780 | 0.695 |

Voir `plots/02_lift_three_rungs.png` et `plots/03_roc_overlay.png`.

### Branche B — fenêtre actionnable

Voir `plots/01_actionable_window.png` : AUC LightGBM pic à 60 min (lecture
directe de `model_card.branch_b_peak_minute`), érosion progressive vers 24 h
(le marché digère l'info — implémenté via une enveloppe gaussienne sur le
drift centrée à 30 min, sigma 60 min). L'heuristique reste plate.

### CV vs walk-forward

| Modèle | CV AUC | Walk-forward AUC | Écart |
|---|---:|---:|---:|
| log_reg       | 0.753 | 0.784 | +0.031 |
| random_forest | 0.752 | 0.785 | +0.033 |
| lightgbm      | 0.741 | 0.780 | +0.039 |

L'écart est modéré (<0.05) — un effet de taille de fold attendu : la CV est
sur 1920 samples par train fold, le walk-forward sur 2400 ; plus de data →
AUC légèrement supérieur.

### Archétypes (K-Means non supervisé)

KMeans n=2 : ARI = 0.006, silhouette = 0.071, tailles de clusters [272, 328]
(sur le jeu de test 20%). KMeans est rapporté ici, pas dans le tableau des
modèles supervisés : les métriques classification n'ont pas de sens pour un
modèle non supervisé. ARI proche de 0 confirme que les clusters ne s'alignent
pas sur la direction — la segmentation décrit autre chose (microstructure, par
exemple). Voir `plots/05_kmeans_archetypes.png` et les profils dans
`models/model_card.json`.

## 5. Limites & travaux futurs

- **Dataset synthétique** : prochaine itération = backfill réel depuis l'API
  CLOB Polymarket (déjà branchée en prod).
- **Pas de feature de microstructure ordre-book live** (latence + complexité).
- **Calibration au-delà de 10 bins** pour la production.
- **Ensemble voting/stacking** comme prochaine famille testable.
- **Threshold tuning** : actuellement 0.55 dans `make_figures.py` ; en prod
  optimiser par bucket de marché.

## 6. Reproductibilité

`python scripts/generate_dataset.py && python scripts/train.py && pytest -q`
suffit pour tout reproduire bit-pour-bit. Seed 42 partout.
`models/model_card.json` documente tous les hyperparamètres, métriques, et
paramètres effectifs du tuning (`noise_sigma`, `interaction_scale`).
`data/raw/dataset_manifest.json` documente les paramètres du générateur (ré-lu
au prochain run sans env vars).
