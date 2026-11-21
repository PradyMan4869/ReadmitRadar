"""
Run-to-run stability check: how much do results move with a different
random train/test patient split?

A single seed's numbers (what scripts/train.py reports) can't answer "is
this stable" — only comparing multiple independent splits can. This
re-derives several train/test splits from the same raw Synthea runs
(same underlying population, only the patient-level split changes),
trains every regime on each, and reports mean +/- std per regime —
plus whether the cross-site generalisation gap's *direction* holds
across splits, since that's the thing the README claims as a finding.

Usage (after scripts/run_synthea.py, needs data_sources/synthea/*):
    python scripts/stability_check.py --seeds 42 7 99 123 2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import REPORTS_DIR
from data.synthea_loader import load_run
from ml.federated import run_comparison
from scripts.prepare_synthea import SYNTHEA_BASE, patient_split

REGIMES = ["local_A", "local_B", "local_A_on_other_site", "local_B_on_other_site",
           "centralized", "federated_personalized_A", "federated_personalized_B",
           "fedavg_logistic"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed stability check")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99, 123, 2024])
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    # Load each hospital's full record set once; only the train/test
    # patient split changes per seed, not the underlying population.
    full = {}
    for hospital, run_name in [("A", "hospital_a"), ("B", "hospital_b")]:
        df, _ = load_run(SYNTHEA_BASE / run_name, hospital)
        full[hospital] = df

    per_seed: dict[str, list] = {r: [] for r in REGIMES}
    cross_site_direction = []  # True if A's local model degrades off-site

    for seed in args.seeds:
        train_dfs, test_parts = {}, []
        for hospital, df in full.items():
            train_df, test_df = patient_split(df, args.test_frac, seed)
            train_dfs[hospital] = train_df
            test_parts.append(test_df)
        test_df = pd.concat(test_parts, ignore_index=True)

        print(f"seed {seed}: training all regimes...")
        results = run_comparison(train_dfs, test_df, n_bootstrap=100)
        for r in REGIMES:
            if r in results:
                per_seed[r].append(results[r]["roc_auc"])

        if "local_A" in results and "local_A_on_other_site" in results:
            cross_site_direction.append(
                results["local_A"]["roc_auc"] > results["local_A_on_other_site"]["roc_auc"]
            )

    summary = {}
    for r, values in per_seed.items():
        if values:
            summary[r] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "values": [round(v, 4) for v in values],
            }

    n_degrades = sum(cross_site_direction)
    n_total = len(cross_site_direction)
    summary["_cross_site_gap_direction_stable"] = {
        "seeds_where_A_degrades_off_site": n_degrades,
        "seeds_tested": n_total,
        "stable": n_degrades in (0, n_total),
    }

    print(f"\n{'regime':<30} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for r in REGIMES:
        if r in summary:
            s = summary[r]
            print(f"{r:<30} {s['mean']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}")
    print(f"\nCross-site gap direction stable across {n_total} seeds? "
          f"{summary['_cross_site_gap_direction_stable']['stable']} "
          f"({n_degrades}/{n_total} seeds show Hospital A degrading off-site)")

    out_path = REPORTS_DIR / "stability_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten -> {out_path}")


if __name__ == "__main__":
    main()
