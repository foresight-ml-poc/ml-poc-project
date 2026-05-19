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
