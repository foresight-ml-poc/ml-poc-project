"""Contract tests for src/config.py."""
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config.py"


def _load_config():
    spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_is_42():
    config = _load_config()
    assert config.SEED == 42


def test_target_column():
    config = _load_config()
    assert config.TARGET_COLUMN == "direction_correct"


def test_features_allowlist_count():
    config = _load_config()
    # 26 named features per brief §3.1 (7+7+5+4+3)
    assert len(config.FEATURES) == 26


def test_buckets():
    config = _load_config()
    assert set(config.BUCKETS) == {"Politics", "Geopolitics", "Crypto", "Economy", "Sports"}


def test_horizon_constants():
    config = _load_config()
    assert config.HORIZON_REF_MIN == 60
    assert config.TRAJECTORY_LEN == 144
    assert config.STEP_MIN == 10
    assert config.TRAJECTORY_LEN * config.STEP_MIN == 24 * 60  # 24 h coverage
    # Pin the derived index so a unilateral change to HORIZON_REF_MIN or STEP_MIN fails loud
    assert config.HORIZON_REF_IDX == config.HORIZON_REF_MIN // config.STEP_MIN
    assert config.HORIZON_REF_IDX == 6


def test_models_registry_keys():
    config = _load_config()
    # Supervised models only in the Basile registry (3 families: linear, bagging, boosting)
    assert set(config.MODELS.keys()) == {"log_reg", "random_forest", "lightgbm"}
    for key, m in config.MODELS.items():
        assert "name" in m and "description" in m and "path" in m


def test_auxiliary_model_paths():
    """logreg_eight (Branch A) and kmeans (archetypes) live OUTSIDE MODELS."""
    config = _load_config()
    assert hasattr(config, "LOGREG_EIGHT_PATH"), "Branch A artifact path missing"
    assert hasattr(config, "KMEANS_PATH"), "Archetype model path missing"


def test_palette_keys():
    config = _load_config()
    expected = {"bg", "card", "line", "mint", "amber", "loss", "ink", "muted"}
    assert set(config.FORESIGHT_PALETTE.keys()) >= expected
