"""Contract tests for the generated dataset and (later) data.py loader."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config.py"
DATA_PATH = PROJECT_ROOT / "src" / "data.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    return _load("project_config", CONFIG_PATH)


def test_signals_csv_shape():
    cfg = _config()
    csv = cfg.RAW_DATA_DIR / "signals_export_sample.csv"
    assert csv.exists(), "Run `python scripts/generate_dataset.py` first."
    df = pd.read_csv(csv)
    assert len(df) == cfg.N_SIGNALS


def test_signals_timestamps_monotonic():
    cfg = _config()
    df = pd.read_csv(cfg.RAW_DATA_DIR / "signals_export_sample.csv")
    ts = pd.to_datetime(df["signal_timestamp"])
    assert ts.is_monotonic_increasing


def test_each_signal_has_trajectory():
    cfg = _config()
    df = pd.read_csv(cfg.RAW_DATA_DIR / "signals_export_sample.csv")
    # Spot-check 50 random rows
    sample = df.sample(50, random_state=42)
    for _, row in sample.iterrows():
        traj_file = cfg.PATHS_DIR / f"{row['signal_id']}.json"
        assert traj_file.exists()
        traj = json.loads(traj_file.read_text())
        assert len(traj["price"]) == cfg.TRAJECTORY_LEN


def test_no_output_columns_in_features():
    """Anti-leakage: FEATURES allowlist must not intersect with OUTPUT_COLUMNS."""
    cfg = _config()
    assert not set(cfg.FEATURES) & set(cfg.OUTPUT_COLUMNS)


def test_dataset_manifest_written():
    """generate_dataset.py persists noise_sigma + interaction_scale for repro."""
    cfg = _config()
    manifest_path = cfg.RAW_DATA_DIR / "dataset_manifest.json"
    assert manifest_path.exists(), "Run `python scripts/generate_dataset.py` first."
    m = json.loads(manifest_path.read_text())
    for key in ["seed", "n_signals", "trajectory_len", "step_min",
                "noise_sigma", "interaction_scale", "generated_at"]:
        assert key in m, f"manifest missing key: {key}"
    assert m["seed"] == 42
    assert 0.5 <= m["noise_sigma"] <= 1.5
    assert 0.5 <= m["interaction_scale"] <= 1.5


def test_target_base_rate():
    """Per brief §3.3: base rate ~52 %, not 50/50 pile.

    Requires data.py to compute the target column from trajectories.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.modules["config"] = _config()
    data = _load("project_data", DATA_PATH)
    df = data._load_processed()
    rate = df[_config().TARGET_COLUMN].mean()
    assert 0.45 <= rate <= 0.60, f"Base rate {rate:.3f} outside [0.45, 0.60]"


def test_split_is_walk_forward():
    """load_dataset_split returns chronological 80/20 — no random shuffle."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.modules["config"] = _config()
    data = _load("project_data", DATA_PATH)
    X_train, X_test, y_train, y_test = data.load_dataset_split()
    cfg = _config()
    n_total = len(y_train) + len(y_test)
    assert n_total == cfg.N_SIGNALS
    # Test size is 20% of total
    assert 0.18 * cfg.N_SIGNALS <= len(y_test) <= 0.22 * cfg.N_SIGNALS
    # Features are 30 columns (25 non-bucket named features + 5 bucket one-hot dummies)
    assert X_train.shape[1] == 30
    assert X_test.shape[1] == 30


def test_no_trajectory_data_in_features():
    """Anti-leakage: scaled X must contain only the 26 allowlist features (post one-hot)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.modules["config"] = _config()
    data = _load("project_data", DATA_PATH)
    cfg = _config()
    # _build_X_y is internal — call via the public split and verify shape
    X_train, X_test, y_train, y_test = data.load_dataset_split()
    # Exactly 30 columns = 25 non-bucket named features + 5 bucket dummies (FEATURES has 26 entries; "bucket" expanded)
    assert X_train.shape[1] == 30
    # Sanity: y is binary 0/1
    import numpy as np
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_horizon_index_maps_to_60min():
    """Pin the trajectory[HORIZON_REF_IDX-1] == price at t=60 min convention."""
    import json
    cfg = _config()
    sample_path = next(cfg.PATHS_DIR.glob("*.json"))
    traj_data = json.loads(sample_path.read_text())
    assert traj_data["step_min"] == 10
    assert len(traj_data["price"]) == cfg.TRAJECTORY_LEN
    # trajectory[i] represents price at t = (i+1) * STEP_MIN minutes.
    # HORIZON_REF_IDX is 1-indexed (step number), so trajectory[HORIZON_REF_IDX-1] is at
    # t = HORIZON_REF_IDX * STEP_MIN minutes = 60 min. Pin this so it can never drift silently.
    assert cfg.HORIZON_REF_IDX == 6
    assert cfg.HORIZON_REF_IDX * cfg.STEP_MIN == 60
