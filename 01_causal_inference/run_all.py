"""
================================================================================
RUN_ALL.PY — Master Pipeline Runner
================================================================================
USAGE:
    uv run python run_all.py

STEPS:
    00 — Load raw Excel + CSV → data/processed/swissgrid_clean_2023_2025.csv
                                  data/processed/entsoe_clean_2023_2025.csv
    01 — Feature engineering  → data/processed/features_train_2023_2024.csv
                                  data/processed/features_val_2025.csv
    02 — Granger causality    → results/granger_results.csv  + GO/NO-GO verdict
    03 — EDA analysis         → results/eda_stats.csv  + supporting CSVs
    04 — Visualisation        → results/causal_validation_swiss_afrr.png

EXPECTED RUNTIMES:
    Step 00: ~2–3 min   (reading 3 large Excel files)
    Step 01: <10 sec
    Step 02: ~1–2 min
    Step 03: ~30 sec
    Step 04: ~1 min

RAW DATA REQUIRED in data/raw/:
    EnergieUebersichtCH-2023.xlsx
    EnergieUebersichtCH-2024.xlsx
    EnergieUebersichtCH-2025.xlsx
    entsoe_swiss_energy_data.csv    ← full 2023–2025 ENTSO-E file
================================================================================
"""

import os, sys, time, importlib.util

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts")

REQUIRED_RAW = [
    "EnergieUebersichtCH-2023.xlsx",
    "EnergieUebersichtCH-2024.xlsx",
    "EnergieUebersichtCH-2025.xlsx",
    "entsoe_swiss_energy_data.csv",
]


def run_step(name: str, script_path: str) -> float:
    print(f"\n{'='*70}\n  RUNNING {name}\n{'='*70}")
    t0   = time.time()
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
    return elapsed


def main():
    raw_dir = os.path.join(SCRIPT_DIR, "..", "data", "raw")

    print("\nSwiss aFRR Causal Inference Pipeline")
    print("="*70)

    missing = [f for f in REQUIRED_RAW
               if not os.path.exists(os.path.join(raw_dir, f))]
    if missing:
        print(f"\n⚠️  Missing files in data/raw/:")
        for f in missing: print(f"      {f}")
        sys.exit(1)
    print("  All input files found ✓")

    steps = [
        ("Step 00 — Load & Preprocess",   "00_load_and_preprocess.py"),
        ("Step 01 — Feature Engineering", "01_feature_engineering.py"),
        ("Step 02 — Granger Causality",   "02_granger_causality.py"),
        ("Step 03 — EDA Analysis",        "03_eda_analysis.py"),
        ("Step 04 — Visualisation",       "04_visualise.py"),
    ]

    t_start = time.time()
    timings = []
    for name, script in steps:
        t = run_step(name, os.path.join(SCRIPTS_DIR, script))
        timings.append((name, t))

    print(f"\n{'='*70}\n  PIPELINE COMPLETE\n{'='*70}")
    for name, t in timings:
        print(f"  {name:<40} {t:>6.1f}s")
    print(f"  {'TOTAL':40} {time.time()-t_start:>6.1f}s")
    print(f"\n  Output: results/causal_validation_swiss_afrr.png\n{'='*70}\n")


if __name__ == "__main__":
    main()
