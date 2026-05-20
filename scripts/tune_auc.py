"""Auto-tune dataset noise/interaction until LightGBM walk-forward AUC ≤ 0.82.

Why: brief §3.3 / §5 cap AUC at ~0.82 for credibility. This script runs the
full generate + train pipeline up to 4 times with progressively stronger
caps, then stops at the first attempt that lands in [0.70, 0.82].

Attempt ladder:
1. default (sigma=0.85, scale=1.0)
2. noise +10% (sigma=0.935, scale=1.0)
3. noise +21% (sigma=1.028, scale=1.0)
4. fallback: noise +10% with interaction scale 0.85 (sigma=0.935, scale=0.85)

If AUC < 0.70, the dataset is over-noisy and the LIGHTGBM AUC IS UNDERFIT;
the script will continue trying the next ladder rung even if AUC is too LOW
(though only AUC > 0.82 typically triggers re-tuning).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import MODEL_CARD_FILE, PROCESSED_DATA_DIR  # noqa: E402

ATTEMPTS = [
    ("default",                 "0.85",  "1.0"),
    ("noise +10%",              "0.935", "1.0"),
    ("noise +21%",              "1.028", "1.0"),
    ("noise +10%, scale 0.85",  "0.935", "0.85"),
]

LO, HI = 0.70, 0.82


def _run(cmd: list[str], env: dict) -> None:
    subprocess.run(cmd, env=env, check=True, cwd=ROOT)


def main() -> int:
    log = []
    cache = PROCESSED_DATA_DIR / "dataset.parquet"
    for i, (label, sigma, scale) in enumerate(ATTEMPTS, start=1):
        env = os.environ.copy()
        env["FORESIGHT_NOISE_SIGMA"] = sigma
        env["FORESIGHT_INTERACTION_SCALE"] = scale
        if cache.exists():
            cache.unlink()
        print(f"\n[attempt {i}/{len(ATTEMPTS)}] {label}  sigma={sigma}  scale={scale}")
        _run([sys.executable, "scripts/generate_dataset.py"], env)
        _run([sys.executable, "scripts/train.py"], env)
        card = json.loads(MODEL_CARD_FILE.read_text())
        auc = card["models"]["lightgbm"]["walk_forward"]["roc_auc"]
        entry = {"attempt": i, "label": label, "noise_sigma": float(sigma),
                 "interaction_scale": float(scale), "lightgbm_auc": auc}
        log.append(entry)
        print(f"  → LightGBM walk-forward AUC = {auc:.3f}")
        if LO <= auc <= HI:
            card["dataset_tuning"] = {**card.get("dataset_tuning", {}), "attempts": log, "final": entry}
            MODEL_CARD_FILE.write_text(json.dumps(card, indent=2))
            print(f"\n✓ AUC in [{LO}, {HI}] — done after {i} attempt(s)")
            return 0

    card["dataset_tuning"] = {**card.get("dataset_tuning", {}), "attempts": log, "final": log[-1], "status": "EXHAUSTED"}
    MODEL_CARD_FILE.write_text(json.dumps(card, indent=2))
    print(f"\n✗ Exhausted all {len(ATTEMPTS)} attempts without landing in [{LO}, {HI}].")
    return 1


if __name__ == "__main__":
    sys.exit(main())
