# Foresight POC — Brief de passation (source spec)

> Ce document est la **source de vérité** du POC. L'arc et les chiffres
> défendables proviennent d'ici. Le plan d'implémentation détaillé est dans
> [docs/superpowers/plans/2026-05-19-foresight-poc.md](docs/superpowers/plans/2026-05-19-foresight-poc.md).

## 0. Cadre

Projet scolaire, POC (proof of concept). Le produit Foresight est réel
(concept, infra, heuristique : voir §1). Le professeur laisse l'étudiant libre
de construire le scénario de démonstration.

Objectif : présentation vendeuse (mélange ML + business) à une classe — poser le
problème, elevator pitch, solution, preuve, avenir. Doit avoir l'air pro et
travaillé en amont.

Méthode imposée : forker https://github.com/basile-desjuzeur/ml-poc-project (le
cours l'exige) → nouveau repo propre, tout construit à neuf dessus. **NE PAS**
repartir de l'ancien repo `ml-foresight`, ni d'aucun ancien dataset / résultat
/ code. Seul élément réutilisé : la **forme visuelle** du deck (voir §2),
recréée dans le fork.

## 1. Le produit (le concept à vendre — réel)

**Foresight** : produit qui surveille l'actualité en temps réel ; dès qu'une
news touche un marché de prédiction Polymarket, il émet un signal (direction
BUY_YES/BUY_NO + score 0–100). Marché de prédiction = on achète « OUI » ou
« NON » à une question ; prix = probabilité ; à la résolution OUI vaut 1 $,
NON 0 $ (ou l'inverse). ~95 % des marchés sont binaires.

**L'heuristique actuelle (formule réelle, faite à la main — la BASELINE) :**

```
signal_strength = 0.15·freshness + 0.10·source_weight + 0.15·confirmation
                + 0.60·(0.65·impact_strength + 0.35·llm_confidence)
trade_quality   = 0.40·liquidity + 0.35·spread + 0.25·time_to_resolution
signal_score    = 0.75·signal_strength + 0.25·trade_quality      (×100 → 0..100)
```

Poids effectifs de chaque entrée dans `signal_score` (pour la Branche A,
tableau « poids main vs poids appris ») :

| Facteur | Poids effectif |
|---|---:|
| freshness | 0.1125 |
| source_weight | 0.0750 |
| confirmation | 0.1125 |
| impact_strength | 0.2925 |
| llm_confidence | 0.1575 |
| liquidity | 0.1000 |
| spread | 0.0875 |
| time_to_resolution | 0.0625 |
| **Somme** | **1.0** |

Le problème (pitch) : l'heuristique est linéaire à poids devinés ; elle ignore
les combinaisons de variables. → c'est là que le ML entre.

Contexte infra réel (storytelling, vrai) : produit en prod sur Hetzner (accès
Tailscale), Postgres dans Docker, pipeline news→LLM→scoring 24/7 ;
reconstruction de prix minute par minute via l'API Polymarket CLOB déjà
réalisée. → l'archi est crédible ; pour le POC, la donnée de démonstration est
représentative/simulée sur cette structure.

## 2. Sur quoi on part (fork Basile, tout à neuf)

LA BASE = un fork propre de https://github.com/basile-desjuzeur/ml-poc-project.
On part de ce squelette vide et on construit tout dessus depuis ce brief.
C'est la méthode imposée par le cours.

NE PAS réutiliser : l'ancien repo `foresight-ml-poc/ml-foresight`, ses
datasets, ses résultats, son code. Rien n'en est repris — tout est régénéré à
neuf (dataset représentatif, pipeline, modèles, figures, README, rapport).

**Seul élément réutilisé : la FORME visuelle du deck.** Le fichier
`foresight-ml-poc/pitch-foresight/index.html` (design sombre/menthe, carte
signal fidèle, navigation clavier, ~18 slides) sert de modèle visuel à recréer
dans le fork (ou un repo deck dédié). On garde la forme, on réécrit 100 % du
fond (cf. §8). Ce n'est pas une dépendance, juste une référence de style.

Conda env Python 3.11. `src/app.py` doit auto-bootstrap son `sys.path`
(`sys.path.insert(0, str(Path(__file__).resolve().parent))` en tête) pour que
`streamlit run src/app.py` marche seul.

## 3. Le dataset représentatif à générer (cœur du POC)

`scripts/generate_dataset.py` → produit, seed fixe (42) :

- `data/raw/signals_export_sample.csv` : 3000 signaux, ordonnés dans le temps
  (timestamps croissants sur ~60 jours) pour le walk-forward.
- `data/raw/paths/<id>.json` : trajectoire de prix par signal, 1 point / 10 min
  sur 24 h = 144 points/signal. C'est l'objet canonique : « on stocke toute la
  trajectoire », pas un snapshot. Une seule granularité partout (= ce que
  consomme la courbe Branche B).

### 3.1 Variables d'ENTRÉE (26 features nommées, jamais le futur)

| Groupe | Features | Distribution |
|---|---|---|
| Sémantique news (7) | impact_strength, llm_confidence, ambiguity_score, specificity_score, cosine_score, novelty_score, sentiment_polarity | Beta sur [0,1] (sentiment [-1,1]) |
| Sources / crédibilité (7) | articles_count, unique_sources_count, tier1_count, tier2_count, tier3_count, source_weight, freshness_min | comptages Poisson ; freshness lognormal |
| Microstructure à l'instant t (5) | market_price_at_signal, bid_ask_spread, liquidity_depth, volatility_pre_24h, time_to_resolution_h | price~Beta ; spread/liq lognormal |
| Contexte (4) | bucket (Politics/Geopolitics/Crypto/Economy/Sports — one-hot), hour_of_day, day_of_week, is_buy_yes | catégoriel / uniforme |
| Dérivées (3) | price_dist_from_0_5 = abs(price-0.5), impact_x_specificity, multi_source_confirmation | calculées |

= **26 features nommées** (le bucket one-hot en ajoute quelques-unes).
Corrélations à injecter : `tier1_count↑ → ambiguity↓` ;
`liquidity_depth↑ → bid_ask_spread↓` ; `impact_strength↑ → |move|↑`.

### 3.2 SORTIES (jamais en entrée — anti-fuite)

L'objet stocké canonique = la **TRAJECTOIRE** (toute la courbe de prix sur
24 h, 1 pt / 10 min = 144 points/signal). PAS de snapshots T+5min / T+30min /
T+1h / etc. — ce découpage en 5 horizons est **SUPPRIMÉ**. Tout est dérivé de
ces 144 points :

- Série dérivée tous les ~10 min : `direction_correct_h ∈ {0,1}` à chaque pas
  de 10 min sur 24 h (≈144 points) — le mouvement signé est-il dans le sens
  prédit à ce pas. → c'est la base de la courbe « fenêtre actionnable » (le
  graphique showpiece lisse). Il n'y a pas de tableau à 5 horizons : la courbe
  à 144 points EST l'analyse de timing.
- Une seule cible binaire de référence `direction_correct = direction_correct_h`
  à un horizon de référence unique (**60 min**) — uniquement parce que le
  contrat Basile (`compute_metrics`, model_card, `load_dataset_split`) exige
  UNE cible binaire. Sert à la Branche A.

Stockage = la trajectoire à 144 pts (1 / 10 min) ; analyse timing = la courbe
AUC sur ces 144 pts ; cible supervisée = 1 SEUL de ces points (60 min,
contrainte Basile). Une seule granularité partout, zéro redondance. **AUCUN**
découpage en 5 horizons.

### 3.3 Le « cerveau » du dataset (rend le ML bénéfique, réaliste)

Probabilité latente de succès :

```
p = sigmoid( β·[ a1·impact·specificity − a2·ambiguity + a3·cosine
                 a4·confirmation + microstructure(spread↓,liquidity↑ favorables)
                 bucket_effect ]
              + horizon_modulation(h)
              + bruit_fort )
```

- **Non-linéaire à interactions** (ex. impact n'aide QUE si specificity haut
  ET ambiguity bas) → une formule linéaire (heuristique) n'en capte qu'une
  partie ; les arbres/boosting captent le reste.
- `horizon_modulation` : edge faible juste après le signal, pic dans la
  fenêtre ~30–120 min, érosion ensuite (le marché digère) → la courbe en
  cloche « fenêtre actionnable » (échantillonnée tous les 10 min, ≈144 pts).
- AUC plafonnée **~0.82** (jamais 0.95 — sinon ça sonne faux).
- Base rate **~52 %** (pas 50/50 pile).
- Un peu de saleté volontaire : ~1–2 % de valeurs manquantes (à imputer), 2–3
  features quasi-inutiles (bruit), walk-forward légèrement < CV. Une data
  trop propre trahit le synthétique.

## 4. Le plan ML (2 branches complémentaires — garder les 2)

### Branche A — Heuristique vs ML (interprétable)

Label = `direction_correct` à l'horizon de référence unique (60 min) — cible
binaire imposée par le contrat Basile (pas un « horizon parmi 5 »).

| Niveau | Quoi | ROC-AUC |
|---|---|---:|
| Baseline | heuristique poids MAIN (8 facteurs, formule §1) | 0.62 |
| ML interprétable | Régression logistique sur les mêmes 8 facteurs (poids appris) | 0.69 |
| ML complet | Gradient Boosting (LightGBM) sur les 26 features | 0.80 |

→ Graphique « lift 3 niveaux » + livrable clé : tableau « poids main vs poids
appris » côte à côte.

### Branche B — La courbe « fenêtre actionnable » (timing produit)

**PAS de tableau à 5 horizons.** On évalue l'AUC (heuristique vs meilleur ML)
sur la série `direction_correct_h` tous les ~10 min sur 24 h (≈144 points) →
une courbe continue en cloche :

- **heuristique** : plate, ~0.55 → ~0.62, sans vrai pic ;
- **ML** : monte vite, pic ~0.82 dans la fenêtre ~60–120 min, puis érosion
  vers ~0.74 à 24 h (le marché digère l'info) ;
- cohérent avec le plafond AUC ~0.82 (§3.3) et l'horizon de référence 60 min
  de la Branche A (~0.80).

Cette courbe à 144 points EST l'analyse de timing — c'est le graphique
showpiece (§6 fig. 1). Message : « le ML ne dit pas seulement SI, mais QUAND
agir : la fenêtre optimale est ~1–2 h après le signal. »

### Modèles (couvre les 3 familles imposées)

1. **Régression logistique** — linéaire (Branche A interprétable)
2. **Random Forest** — ensemble d'arbres / bagging
3. **Gradient Boosting (LightGBM)** (ou XGBoost) — modèle héros
4. **K-Means** — non supervisé → archétypes de signaux (segmentation)
5. + **SHAP** (explicabilité) + courbe de calibration

Optionnel : ensemble voting/stacking comme modèle final.

### Méthodo (ce qui fait « pro » — non négociable)

- Anti-fuite par allowlist explicite : seules les 26 features signal-time
  entrent dans X (jamais les `move_*` / labels).
- 5-fold CV + walk-forward strict (train passé → test futur) ; le walk-forward
  doit être légèrement < CV (réaliste).
- Calibration + seuil optimisé (courbe précision/rappel), pas 0.5 naïf.
- Model card + reproductibilité (seed). `pytest -q` doit passer.

## 5. Chiffres cibles (défendables — NE PAS gonfler, cohérents avec §4)

**0.80 = ROC-AUC, PAS un winrate.** Trois métriques distinctes :

|                              | ROC-AUC | Accuracy | Winrate (NET de spread) |
|------------------------------|--------:|---------:|------------------------:|
| Heuristique (poids main)     |    0.62 |     0.60 |                  ~54 % |
| Régression log. (poids appris) | 0.69 |     0.66 |                  ~57 % |
| LightGBM (héros, @60 min réf.) | 0.80 |     0.73 |              ~60–61 % |

Discours : « +18 pts d'AUC, +6 pts de winrate net de coûts — modeste mais réel ;
en trading c'est la différence entre perdre et une espérance positive. » Winrate
plafonné ~60 % (jamais 70 %+ : crédibilité détruite). AUC plafonnée ~0.82. AUC
en avant, winrate sobre et net de spread.

## 6. Les graphiques (très beaux — charte Foresight, faits-main)

**Tokens :** fond `#070a0f`, carte `#0c1014`, lignes `#252e3d`, menthe
`#0BE0A6`, ambre `#f5b942`, perte `#f76d6d`, encre `#eef3f9`, atténué
`#9aa7b8`. Police DejaVu Sans (matplotlib). **Bande d'en-tête réservée**
(titre + sous-titre placés AU-DESSUS de la zone via `add_axes` — jamais de
chevauchement titre/graphe). 150 dpi, gros titres, annotations.

`scripts/make_figures.py` → **6 figures premium** :

1. **Courbe en cloche « fenêtre actionnable »** (showpiece, 144 pts)
2. **Lift 3 niveaux** : poids main → poids appris → ML complet
3. **ROC overlay** : heuristique vs les modèles
4. **SHAP beeswarm** (impact des features — l'effet waouh)
5. **Archétypes K-Means** (scatter 2D PCA, coloré)
6. **Backtest equity/winrate net de spread + matrice de confusion + courbe de calibration**

## 7. Structure Basile (contrats à respecter)

```
src/config.py     # PATHS, SEED=42, TARGET_COLUMN, FEATURE allowlist, MODELS{}
src/data.py       # _clean(), _feature_engineer(), load_dataset_split()
                  #   -> (X_train,X_test,y_train,y_test) numpy scalés (fit train only)
src/metrics.py    # compute_metrics(y_true,y_pred) -> dict(acc,prec,rec,f1,roc_auc)
src/app.py        # build_app() Streamlit (auto-bootstrap sys.path en tête)
src/model_io.py   src/results.py   src/__init__.py     # FIXÉS Basile, ne pas toucher
scripts/main.py   # FIXÉ Basile : eval des MODELS sur test + lance Streamlit
scripts/train.py            # 5 modèles + Branche A + B + model_card
scripts/generate_dataset.py # dataset 3000 + trajectoires 144 pts (NOUVEAU)
scripts/make_figures.py     # les 6 graphes premium
scripts/honest_analysis.py  # consolide results/*.json -> synthèse
models/model_card.json + *.joblib (gitignorés sauf card)
plots/ (figures commitées)   results/*.json   docs/rapport.md
```

`config.MODELS` liste `logreg/random_forest/lightgbm/kmeans` avec chemins
`.joblib` ; chaque modèle expose `.predict` (KMeans sklearn natif, **PAS de
classe custom** — sinon `main.py` ne peut pas l'unpickler).

> **Note d'architecture (post-feedback).** Le plan d'implémentation a
> ajusté ce point : `config.MODELS` ne contient que les **3 modèles
> supervisés** (LogReg, RF, LightGBM). Le `logreg_eight` (Branche A) et
> `kmeans` (archétypes) vivent comme artefacts **auxiliaires** hors registry
> — cf. plan §Task 2 et §Self-review. KMeans rapporté avec métriques
> non supervisées (ARI + silhouette + profils), pas avec des métriques
> classification.

## 8. Le deck (réécrire le fond, GARDER la forme `pitch-foresight/index.html`)

**Arc :** Cover → C'est quoi un marché prédictif → L'idée → Le problème
(heuristique plafonne) → La solution (le ML) → Comment ça marche → PREUVE
Branche A (lift 3 niveaux) → PREUVE Branche B (courbe fenêtre actionnable) →
SHAP/insight produit → Démo carte signal → Avenir/roadmap → Conclusion +
elevator pitch → 3 annexes ML (anti-fuite, CV vs walk-forward, modèles).

Inclure la phrase de cadrage du §0 (telle quelle). Elevator pitch 30 s en intro
ET conclusion. Graphes faits-main SVG/CSS dans le deck (pas de captures),
charte du §6.

## 9. Checklist livrables & commandes

```bash
conda create -n poc-foresight python=3.11 -y && conda activate poc-foresight
pip install -r requirements.txt
python scripts/generate_dataset.py     # 3000 signaux + trajectoires 144 pts (seed 42)
python scripts/train.py                # 5 modèles + Branche A + B + model_card
python scripts/honest_analysis.py      # synthèse results/*.json
python scripts/make_figures.py         # 6 figures premium dans plots/
pytest -q                              # doit passer
python scripts/main.py                 # contrat Basile + Streamlit (8501)
# deck : open pitch-foresight/index.html
```

## 10. Liens

- À FORKER (la base, méthode imposée) :
  https://github.com/basile-desjuzeur/ml-poc-project
- Forme du deck à recréer (style only) :
  https://github.com/foresight-ml-poc/pitch-foresight (`index.html`)
- Org (pour pousser le fork) : https://github.com/foresight-ml-poc

⚠️ **Ancien repo `foresight-ml-poc/ml-foresight` : NE PAS réutiliser** (ni
son code, ni ses datasets, ni ses résultats). Référence historique
uniquement.
