"""Project configuration: paths, seed, feature allowlist, palette, model registry."""
from pathlib import Path

# --- Paths (Basile contract — keep section intact) ---
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PATHS_DIR = RAW_DATA_DIR / "paths"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

for _dir in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PATHS_DIR,
    LOGS_DIR, MODELS_DIR, NOTEBOOKS_DIR, PLOTS_DIR, RESULTS_DIR,
    SCRIPTS_DIR, TESTS_DIR, DOCS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# --- Reproducibility ---
SEED = 42

# --- Dataset ---
N_SIGNALS = 3000
TRAJECTORY_LEN = 144      # 1 point / 10 min over 24 h
STEP_MIN = 10
HORIZON_REF_MIN = 60      # Branch A reference horizon (60 min)
HORIZON_REF_IDX = HORIZON_REF_MIN // STEP_MIN  # = 6 (index into 144-pt trajectory)
TIMESTAMP_SPAN_DAYS = 60  # signals ordered over ~60 days

TARGET_COLUMN = "direction_correct"   # binary, computed at HORIZON_REF_MIN

BUCKETS = ["Politics", "Geopolitics", "Crypto", "Economy", "Sports"]

# 26 named features (brief §3.1). Bucket gets one-hot expanded downstream
# but the *named* feature is "bucket".
FEATURES = [
    # Sémantique news (7)
    "impact_strength", "llm_confidence", "ambiguity_score", "specificity_score",
    "cosine_score", "novelty_score", "sentiment_polarity",
    # Sources / crédibilité (7)
    "articles_count", "unique_sources_count", "tier1_count", "tier2_count",
    "tier3_count", "source_weight", "freshness_min",
    # Microstructure (5)
    "market_price_at_signal", "bid_ask_spread", "liquidity_depth",
    "volatility_pre_24h", "time_to_resolution_h",
    # Contexte (4)
    "bucket", "hour_of_day", "day_of_week", "is_buy_yes",
    # Dérivées (3)
    "price_dist_from_0_5", "impact_x_specificity", "multi_source_confirmation",
]
assert len(FEATURES) == 26, "FEATURES allowlist must contain exactly 26 named features"

# Output-only columns (NEVER allowed into X — anti-leakage allowlist enforced in data.py)
OUTPUT_COLUMNS = [
    "direction_correct",     # target at HORIZON_REF_MIN
    "signal_id",
    "signal_timestamp",
    "trajectory_path",       # filename of paths/<id>.json
    "predicted_direction",   # is_buy_yes already, but stored separately for clarity
]

# 8-factor heuristic (brief §1) — names refer to FEATURES (some need normalization)
HEURISTIC_8_FACTORS = [
    "freshness_min",            # → normalized 1/(1+freshness_min/30)  (fresh = high)
    "source_weight",            # already [0,1]
    "multi_source_confirmation",# already [0,1]
    "impact_strength",          # [0,1]
    "llm_confidence",           # [0,1]
    "liquidity_depth",          # → normalized log scale [0,1]
    "bid_ask_spread",           # → normalized inverse (tight = high quality)
    "time_to_resolution_h",     # → normalized (mid = good, far = degraded)
]

# Effective weights of each input in signal_score (brief §1)
HEURISTIC_WEIGHTS = {
    "freshness_min": 0.1125,
    "source_weight": 0.0750,
    "multi_source_confirmation": 0.1125,
    "impact_strength": 0.2925,
    "llm_confidence": 0.1575,
    "liquidity_depth": 0.1000,
    "bid_ask_spread": 0.0875,
    "time_to_resolution_h": 0.0625,
}
assert abs(sum(HEURISTIC_WEIGHTS.values()) - 1.0) < 1e-9, "Heuristic weights must sum to 1.0"

# --- Foresight palette (brief §6) ---
FORESIGHT_PALETTE = {
    "bg": "#070a0f",
    "card": "#0c1014",
    "line": "#252e3d",
    "mint": "#0BE0A6",
    "amber": "#f5b942",
    "loss": "#f76d6d",
    "ink": "#eef3f9",
    "muted": "#9aa7b8",
}

# --- Model registry (Basile contract) ---
# Three SUPERVISED families: linear (LogReg), bagging (RandomForest),
# gradient boosting (LightGBM). All trained on the same 30-column scaled
# feature matrix (25 numeric + 5 bucket one-hot). Like-for-like comparable.
MODELS = {
    "log_reg": {
        "name": "Logistic Regression",
        "description": "Linear baseline on the standardized 30-column feature matrix.",
        "path": MODELS_DIR / "log_reg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Bagging ensemble (400 trees, max_depth 12) on the same feature matrix.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "lightgbm": {
        "name": "LightGBM (hero)",
        "description": "Gradient Boosting on the same feature matrix. Captures non-linear interactions the heuristic misses.",
        "path": MODELS_DIR / "lightgbm.joblib",
    },
}

# --- Auxiliary artifacts (NOT in the Basile registry) ---
# Branch A only: LogReg trained on the 8 normalized heuristic factors. Used
# to build the "hand vs learned weights" table and the 3-rung lift chart.
LOGREG_EIGHT_PATH = MODELS_DIR / "logreg_eight.joblib"

# Archetype analysis only: KMeans n=2. Reported with unsupervised metrics
# (ARI vs target, silhouette, cluster sizes, archetype profiles) — NOT in
# the supervised metrics table.
KMEANS_PATH = MODELS_DIR / "kmeans.joblib"
