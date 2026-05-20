"""Contract tests for trained models, model_card.json, and Branch A/B outputs."""
import json
import importlib.util
from pathlib import Path
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config.py"


def _config():
    spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_card_present():
    card = _config().MODEL_CARD_FILE
    assert card.exists(), "Run `python scripts/train.py` first."


def test_all_models_have_predict():
    cfg = _config()
    for key, m in cfg.MODELS.items():
        obj = joblib.load(m["path"])
        assert hasattr(obj, "predict"), f"{key} missing .predict"


def test_lightgbm_auc_cap_respected():
    """Brief §5: AUC ≤ ~0.82 on walk-forward test."""
    card = json.loads(_config().MODEL_CARD_FILE.read_text())
    auc = card["models"]["lightgbm"]["walk_forward"]["roc_auc"]
    assert 0.70 <= auc <= 0.82, f"LightGBM AUC {auc} outside [0.70, 0.82]"


def test_walkforward_not_far_above_cv():
    """Brief §3.3 & §5 intent: walk-forward in the same ballpark as CV.

    Note: with IID-sampled synthetic data, CV folds use less train data (1920) than the
    walk-forward fit (2400), so WF AUC naturally edges above CV by ~0.03. Tolerance is
    set to +0.05 to keep the test meaningful without flagging this size-effect artifact.
    """
    card = json.loads(_config().MODEL_CARD_FILE.read_text())
    for key in ["log_reg", "random_forest", "lightgbm"]:
        m = card["models"][key]
        assert m["walk_forward"]["roc_auc"] <= m["cv"]["roc_auc"] + 0.05, (
            f"{key} walk-forward AUC ({m['walk_forward']['roc_auc']:.3f}) "
            f"exceeds CV AUC ({m['cv']['roc_auc']:.3f}) by more than 0.05 — "
            "suggests dataset has a strong time-trend, which is unexpected with IID sampling"
        )


def test_branch_a_lift_present():
    cfg = _config()
    p = cfg.RESULTS_DIR / "branch_a_lift.json"
    assert p.exists()
    data = json.loads(p.read_text())
    for rung in ["heuristic_hand", "logreg_8_learned", "lightgbm_26"]:
        assert rung in data
        assert "roc_auc" in data[rung]
    assert data["heuristic_hand"]["roc_auc"] < data["logreg_8_learned"]["roc_auc"]
    assert data["logreg_8_learned"]["roc_auc"] < data["lightgbm_26"]["roc_auc"]


def test_branch_b_curve_144_points():
    cfg = _config()
    curve = json.loads((cfg.RESULTS_DIR / "branch_b_curve.json").read_text())
    assert len(curve["heuristic_auc"]) == 144
    assert len(curve["lightgbm_auc"]) == 144
    peak_idx = int(np.argmax(curve["lightgbm_auc"]))
    assert 3 <= peak_idx <= 24  # 30-240 min window


def test_archetypes_reported_with_unsupervised_metrics():
    card = json.loads(_config().MODEL_CARD_FILE.read_text())
    arch = card["archetypes"]
    assert "ari" in arch and "silhouette" in arch
    assert -1.0 <= arch["ari"] <= 1.0
    assert -1.0 <= arch["silhouette"] <= 1.0
    assert "cluster_sizes" in arch and sum(arch["cluster_sizes"]) > 0
    assert "profiles" in arch and arch["profiles"]


def test_logreg_eight_artifact_present():
    p = _config().LOGREG_EIGHT_PATH
    assert p.exists(), "logreg_eight.joblib not saved by train.py"
    lr8 = joblib.load(p)
    assert hasattr(lr8, "predict_proba")
    assert lr8.coef_.shape[1] == 8


def test_dataset_tuning_logged():
    card = json.loads(_config().MODEL_CARD_FILE.read_text())
    tuning = card["dataset_tuning"]
    assert "noise_sigma" in tuning and "interaction_scale" in tuning
    assert 0.3 <= tuning["noise_sigma"] <= 1.5
    assert 0.3 <= tuning["interaction_scale"] <= 1.5
