"""
================================================================================
run_all.py — Master Runner for 02_model_training
================================================================================
Runs all 4 steps in sequence:
    Step 00 — Prepare & validate features
    Step 01 — Train XGBoost
    Step 02 — Evaluate on validation set
    Step 03 — XAI / SHAP analysis

Usage:
    uv run python 02_model_training/run_all.py
================================================================================
"""

import subprocess
import sys
import time
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")

STEPS = [
    ("00", "00_prepare_features.py", "Prepare & Validate Features"),
    ("01", "01_train_xgboost.py",    "Train XGBoost"),
    ("02", "02_evaluate.py",         "Evaluate on Validation Set"),
    ("03", "03_explain.py",          "XAI / SHAP Analysis"),
]


def run_step(step_id: str, script: str, name: str) -> bool:
    path = os.path.join(SCRIPTS_DIR, script)
    print(f"\n{'='*60}")
    print(f"STEP {step_id} — {name}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, path], check=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n❌ STEP {step_id} FAILED after {elapsed:.0f}s")
        return False
    print(f"\n✅ STEP {step_id} complete ({elapsed:.0f}s)")
    return True


def main():
    total_start = time.time()
    print("=" * 60)
    print("02_model_training — Full Pipeline")
    print("=" * 60)

    for step_id, script, name in STEPS:
        if not run_step(step_id, script, name):
            print(f"\n❌ Pipeline stopped at Step {step_id}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ ALL STEPS COMPLETE ({total_elapsed/60:.1f} minutes)")
    print(f"   View results: uv run mlflow ui --backend-store-uri mlruns")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()