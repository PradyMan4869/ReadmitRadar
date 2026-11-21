"""Federated training: FedAvg math, histogram-aggregation XGBoost, and
regime comparison behaviour."""
import numpy as np
import pytest

from ml.federated import (
    FedAvgLogistic, FederatedHistogramXGBoost, run_comparison,
)
from ml.features import FEATURE_COLUMNS, TARGET
from ml.model import ReadmissionModel

FAST_XGB = {"n_estimators": 60, "max_depth": 3}
FAST_FED_HIST = {"n_estimators": 40, "max_depth": 3}


@pytest.fixture(scope="module")
def results(hospital_a, hospital_b, test_set):
    return run_comparison(
        {"A": hospital_a, "B": hospital_b}, test_set,
        fedavg_rounds=5, xgb_params=FAST_XGB,
        fed_hist_params=FAST_FED_HIST, n_bootstrap=100,
    )


def test_all_regimes_present(results):
    for key in ("local_A", "local_B", "centralized",
                "federated_personalized_A", "federated_personalized_B",
                "fedavg_logistic"):
        assert key in results, key


def test_every_regime_has_confidence_interval(results):
    for key, m in results.items():
        assert "roc_auc_ci" in m and "pr_auc_ci" in m, key
        lo, hi = m["roc_auc_ci"]
        assert lo <= m["roc_auc"] <= hi, f"{key}: {m['roc_auc']} not in [{lo}, {hi}]"


def test_models_learn_signal(results):
    # Every deployable regime must beat chance decisively; the
    # *_on_other_site entries are diagnostics that are expected to degrade
    for key, m in results.items():
        floor = 0.55 if key.endswith("_on_other_site") else 0.60
        assert m["roc_auc"] > floor, f"{key}: {m['roc_auc']}"


def test_cross_site_generalisation_gap_exists(results):
    # A model trained on one hospital should be measurably worse on the
    # other site than on the pooled test set — the premise of the project
    if "local_A_on_other_site" in results:
        assert (results["local_A"]["roc_auc"]
                > results["local_A_on_other_site"]["roc_auc"])


def test_federated_personalized_close_to_local(results):
    # Shared splits should be in the same ballpark as that hospital's own
    # local-only model scored on the same site's rows — not identical (the
    # small fixture + fewer boosting rounds here means neither regime is
    # fully converged), just not a different model entirely.
    for name in ("A", "B"):
        local = results[f"local_{name}"]["roc_auc"]
        fed = results[f"federated_personalized_{name}"]["roc_auc"]
        assert fed > local - 0.12, f"{name}: fed {fed} vs local {local}"


def test_fedavg_weight_averaging_shape(hospital_a, hospital_b):
    fed = FedAvgLogistic(rounds=3, local_epochs=2).fit([hospital_a, hospital_b])
    n_features = len(hospital_a.columns) - 3  # minus id/hospital/target
    assert fed.weights.shape == (n_features + 1,)  # + intercept
    probs = fed.predict_proba(hospital_a)
    assert np.all((probs >= 0) & (probs <= 1))


def test_federated_histogram_split_matches_pooled_brute_force(hospital_a, hospital_b):
    # The core claim: the FIRST split chosen from cross-hospital-summed
    # histograms must match the split a pooled-data model would choose —
    # i.e. histogram aggregation is mathematically equivalent to having
    # all rows in one place, not an approximation.
    import pandas as pd
    from ml.federated import _Histogram, _quantile_bin_edges, _split_gain

    dfs = [hospital_a, hospital_b]
    edges = _quantile_bin_edges(dfs, n_bins=64)
    n_bins = max(len(e) for e in edges.values())

    def bin_df(df):
        return {
            col: np.clip(np.searchsorted(edges[col], df[col].values, side="right") - 1,
                         0, len(edges[col]) - 1)
            for col in FEATURE_COLUMNS
        }

    pooled = pd.concat(dfs, ignore_index=True)
    y = pooled[TARGET].values.astype(float)
    base = np.log(y.mean() / (1 - y.mean()))
    grad_pooled = (1 / (1 + np.exp(-base))) - y
    hess_pooled = np.full(len(y), (1 / (1 + np.exp(-base))) * (1 - 1 / (1 + np.exp(-base))))
    pooled_binned = bin_df(pooled)
    pooled_hist = _Histogram.build(pooled_binned, grad_pooled, hess_pooled,
                                    np.ones(len(y), dtype=bool), n_bins)

    per_client_hist = []
    for df in dfs:
        yc = df[TARGET].values.astype(float)
        gc = (1 / (1 + np.exp(-base))) - yc
        hc = np.full(len(yc), (1 / (1 + np.exp(-base))) * (1 - 1 / (1 + np.exp(-base))))
        per_client_hist.append(_Histogram.build(bin_df(df), gc, hc,
                                                  np.ones(len(yc), dtype=bool), n_bins))
    federated_hist = _Histogram.sum_across_clients(per_client_hist)

    def best_split(hist):
        best = (0.0, None, None)
        for col in hist.grad:
            g_cum, h_cum = np.cumsum(hist.grad[col]), np.cumsum(hist.hess[col])
            g_total, h_total = g_cum[-1], h_cum[-1]
            for b in range(n_bins - 1):
                gain = _split_gain(g_cum[b], h_cum[b], g_total - g_cum[b],
                                    h_total - h_cum[b], lam=1.0)
                if gain > best[0]:
                    best = (gain, col, b)
        return best[1], best[2]

    assert best_split(pooled_hist) == best_split(federated_hist)


def test_federated_histogram_leaves_are_personalized(hospital_a, hospital_b, test_set):
    fed = FederatedHistogramXGBoost(**FAST_FED_HIST).fit([hospital_a, hospital_b])
    probs_a = fed.predict_proba(test_set, client_index=0)
    probs_b = fed.predict_proba(test_set, client_index=1)
    assert probs_a.shape == probs_b.shape == (len(test_set),)
    assert np.all((probs_a > 0) & (probs_a < 1))
    # Same tree structure, different leaf values per hospital -> predictions differ
    assert not np.allclose(probs_a, probs_b)


def test_unfitted_raises(test_set):
    with pytest.raises(RuntimeError):
        FederatedHistogramXGBoost().predict_proba(test_set, client_index=0)
    with pytest.raises(RuntimeError):
        FedAvgLogistic().predict_proba(test_set)
