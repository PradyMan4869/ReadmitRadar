"""Parse the Synthea runs into training datasets.

Reads data_sources/synthea/hospital_{a,b}/csv, derives discharge records
with 30-day readmission labels, and writes storage/hospital_a.csv,
hospital_b.csv and test.csv (the same files scripts/train.py consumes).

The test set is split **by patient** (never by row) so no patient appears
in both train and test.

Usage (after scripts/run_synthea.py):
    python scripts/prepare_synthea.py [--test-frac 0.2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import DATA_DIR, REPORTS_DIR
from data.synthea_loader import load_run
from ml.features import TARGET

SYNTHEA_BASE = Path(__file__).resolve().parent.parent / "data_sources" / "synthea"


def patient_split(df: pd.DataFrame, test_frac: float, seed: int):
    """Split admissions by patient so test patients are fully unseen."""
    patients = df["patient_id"].str.split("-").str[1]
    unique = patients.unique()
    rng = np.random.default_rng(seed)
    test_patients = set(rng.choice(unique, size=int(len(unique) * test_frac),
                                   replace=False))
    mask = patients.isin(test_patients)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build datasets from Synthea runs")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_stats = {}
    test_parts = []
    for hospital, run_name in [("A", "hospital_a"), ("B", "hospital_b")]:
        df, stats = load_run(SYNTHEA_BASE / run_name, hospital)
        all_stats[run_name] = stats
        train_df, test_df = patient_split(df, args.test_frac, args.seed)
        train_df.to_csv(DATA_DIR / f"{run_name}.csv", index=False)
        test_parts.append(test_df)
        print(f"{run_name}: {stats['admissions']} admissions from "
              f"{stats['patients']} patients | readmit rate "
              f"{stats['readmission_rate']:.1%} | train {len(train_df)} / "
              f"test {len(test_df)}")
        print(f"  lab imputation %: {stats['lab_imputed_pct']}")

    test = (pd.concat(test_parts, ignore_index=True)
            .sample(frac=1, random_state=args.seed).reset_index(drop=True))
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"test.csv: {len(test)} admissions | readmit rate "
          f"{test[TARGET].mean():.1%}")

    (REPORTS_DIR / "synthea_dataset_stats.json").write_text(
        json.dumps(all_stats, indent=2), encoding="utf-8")
    print("Next: python scripts/train.py")


if __name__ == "__main__":
    main()
