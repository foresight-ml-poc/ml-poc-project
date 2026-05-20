"""Print a single consolidated summary from all results/*.json files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import MODEL_CARD_FILE, RESULTS_DIR, STEP_MIN


def _fmt_metrics(m: dict) -> str:
    keys = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    return "  ".join(f"{k}={m[k]:.3f}" for k in keys if k in m)


def main() -> None:
    card = json.loads(MODEL_CARD_FILE.read_text())
    branch_a = json.loads((RESULTS_DIR / "branch_a_lift.json").read_text())
    branch_b = json.loads((RESULTS_DIR / "branch_b_curve.json").read_text())

    print("=" * 72)
    print(f"FORESIGHT POC — HONEST ANALYSIS  (seed={card['seed']})")
    print("=" * 72)
    print(f"Signals: {card['n_signals']}  |  Features: {card['n_features']}  "
          f"|  Horizon ref: {card['horizon_ref_min']} min")
    print(f"Split: {card['split']}")
    tuning = card.get("dataset_tuning", {})
    print(f"Dataset tuning: noise_sigma={tuning.get('noise_sigma')}  "
          f"interaction_scale={tuning.get('interaction_scale')}")
    print()

    print("--- BRANCH A: heuristic → LogReg-8 → LightGBM-26 (60-min target) ---")
    for rung in ["heuristic_hand", "logreg_8_learned", "lightgbm_26"]:
        print(f"  {rung:>20}  {_fmt_metrics(branch_a[rung])}")
    print()

    print("--- CV vs walk-forward AUC ---")
    for key, m in card["models"].items():
        cv = m["cv"]["roc_auc"]
        wf = m["walk_forward"]["roc_auc"]
        flag = "OK" if wf <= cv + 0.05 else "WARN"
        print(f"  {key:>14}  CV={cv:.3f}  WF={wf:.3f}  [{flag}]")
    print()

    peak = int(card["branch_b_peak_idx"])
    peak_min = card["branch_b_peak_minute"]
    print(f"--- BRANCH B: actionable window ---")
    print(f"  Heuristic AUC range: "
          f"[{min(branch_b['heuristic_auc']):.3f}, {max(branch_b['heuristic_auc']):.3f}]")
    print(f"  LightGBM  AUC range: "
          f"[{min(branch_b['lightgbm_auc']):.3f}, {max(branch_b['lightgbm_auc']):.3f}]")
    print(f"  Peak window: step {peak + 1} = {peak_min} min after signal")
    print()

    print("--- ARCHETYPES (K-Means, unsupervised) ---")
    arch = card.get("archetypes", {})
    print(f"  ARI vs target: {arch.get('ari', 0):.4f}  (≈0 means clusters ≠ direction)")
    print(f"  Silhouette:    {arch.get('silhouette', 0):.4f}")
    print(f"  Cluster sizes: {arch.get('cluster_sizes', [])}")
    print()

    print("--- Branch A weights: HAND vs LEARNED (8 factors) ---")
    hand = branch_a["heuristic_hand"]["weights"]
    learned = branch_a["logreg_8_learned"]["weights"]
    print(f"  {'factor':<28}{'hand':>10}{'learned':>12}")
    for f in hand:
        print(f"  {f:<28}{hand[f]:>10.4f}{learned[f]:>12.4f}")


if __name__ == "__main__":
    main()
