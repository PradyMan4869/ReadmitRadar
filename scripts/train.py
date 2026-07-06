"""Train all regimes (local / centralized / federated), write metrics + chart.

Run after scripts/generate_data.py:
    python scripts/train.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, TRAIN
from ml.federated import run_comparison
from ml.model import ReadmissionModel, save_metrics

# Chart styling (validated reference palette; single-hue + emphasis)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
BLUE = "#2a78d6"        # federated regimes (the story)
GRAY = "#c9c8c4"        # baselines

REGIME_ORDER = [
    ("local_A", "Local — Hospital A only", GRAY),
    ("local_B", "Local — Hospital B only", GRAY),
    ("fedavg_logistic", "FedAvg (logistic)", BLUE),
    ("federated_xgboost", "Federated XGBoost", BLUE),
    ("centralized", "Centralized (PHI pooled)", GRAY),
]


def load_datasets():
    paths = {name: DATA_DIR / f"{name}.csv"
             for name in ("hospital_a", "hospital_b", "test")}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing {missing} in {DATA_DIR} — run scripts/generate_data.py first."
        )
    return {k: pd.read_csv(v) for k, v in paths.items()}


def plot_comparison(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]
    rows = [(label, color, results[key]) for key, label, color in REGIME_ORDER
            if key in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes, metrics):
        labels = [r[0] for r in rows][::-1]
        colors = [r[1] for r in rows][::-1]
        values = [r[2][metric] for r in rows][::-1]

        ax.set_facecolor(SURFACE)
        bars = ax.barh(labels, values, color=colors, height=0.55)
        for bar, v in zip(bars, values):
            ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9, color=INK)
        lo = min(values)
        ax.set_xlim(max(0.0, lo - 0.05), max(values) + 0.045)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.grid(axis="x", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("30-day readmission — training regimes on pooled held-out set",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_datasets()
    hospital_dfs = {"A": data["hospital_a"], "B": data["hospital_b"]}

    print("Training local / centralized / federated regimes...")
    results = run_comparison(hospital_dfs, data["test"],
                             fedavg_rounds=TRAIN.fedavg_rounds)

    print(f"\n{'regime':<28} {'ROC-AUC':>8} {'PR-AUC':>8} {'recall':>8}")
    for key, m in results.items():
        print(f"{key:<28} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
              f"{m['recall']:>8.4f}")

    save_metrics(results, REPORTS_DIR / "federated_metrics.json")

    chart = REPORTS_DIR / "regime_comparison.png"
    plot_comparison(results, chart)
    print(f"\nChart -> {chart}")

    # The deployable model is the federated XGBoost ensemble; persist its
    # members plus a centralized reference model for the UI's SHAP explainer.
    for name, df in hospital_dfs.items():
        ReadmissionModel().fit(df).save(MODELS_DIR / f"xgb_hospital_{name}.json")
    pooled = pd.concat(hospital_dfs.values(), ignore_index=True)
    # Calibrated so predict_proba is a true probability (matches the actual
    # base rate) — the UI/demo display this as "risk %", so it must be
    # calibrated even though scale_pos_weight (used for the regime-comparison
    # metrics above) skews raw probabilities for better ranking/recall.
    ReadmissionModel(calibrate=True).fit(pooled).save(MODELS_DIR / "xgb_reference.json")
    print(f"Models -> {MODELS_DIR}")

    print(json.dumps({"summary": {
        k: results[k]["roc_auc"] for k, _, _ in REGIME_ORDER if k in results
    }}, indent=2))


if __name__ == "__main__":
    main()
