"""Federated training: FedAvg math and regime comparison behaviour."""
import numpy as np
import pytest

from ml.federated import FedAvgLogistic, FederatedXGBoost, run_comparison
from ml.features import TARGET

FAST_XGB = {"n_estimators": 60, "max_depth": 3}


@pytest.fixture(scope="module")
def results(hospital_a, hospital_b, test_set):
    return run_comparison(
        {"A": hospital_a, "B": hospital_b}, test_set,
        fedavg_rounds=5, xgb_params=FAST_XGB,
    )


def test_all_regimes_present(results):
    for key in ("local_A", "local_B", "centralized",
                "federated_xgboost", "fedavg_logistic"):
        assert key in results, key


def test_models_learn_signal(results):
    # Every deployable regime must beat chance decisively; the
    # *_on_other_site entries are diagnostics that are expected to degrade
    for key, m in results.items():
        floor = 0.55 if key.endswith("_on_other_site") else 0.65
        assert m["roc_auc"] > floor, f"{key}: {m['roc_auc']}"


def test_cross_site_generalisation_gap_exists(results):
    # A model trained on one hospital should be measurably worse on the
    # other site than on the pooled test set — the premise of the project
    if "local_A_on_other_site" in results:
        assert (results["local_A"]["roc_auc"]
                > results["local_A_on_other_site"]["roc_auc"])


def test_federated_beats_worst_local(results):
    worst_local = min(results["local_A"]["roc_auc"],
                      results["local_B"]["roc_auc"])
    assert results["federated_xgboost"]["roc_auc"] >= worst_local


def test_federated_close_to_centralized(results):
    # The federated model should recover most of the pooled-data benefit
    gap = results["centralized"]["roc_auc"] - results["federated_xgboost"]["roc_auc"]
    assert gap < 0.05, f"federated trails centralized by {gap:.3f}"


def test_fedavg_weight_averaging_shape(hospital_a, hospital_b):
    fed = FedAvgLogistic(rounds=3, local_epochs=2).fit([hospital_a, hospital_b])
    n_features = len(hospital_a.columns) - 3  # minus id/hospital/target
    assert fed.weights.shape == (n_features + 1,)  # + intercept
    probs = fed.predict_proba(hospital_a)
    assert np.all((probs >= 0) & (probs <= 1))


def test_federated_xgboost_margin_average(hospital_a, hospital_b, test_set):
    fed = FederatedXGBoost(FAST_XGB).fit([hospital_a, hospital_b])
    probs = fed.predict_proba(test_set)
    assert probs.shape == (len(test_set),)
    assert np.all((probs > 0) & (probs < 1))
    # Aggregate must not simply equal either member
    member = fed.members[0].predict_proba(test_set)
    assert not np.allclose(probs, member)


def test_unfitted_raises(test_set):
    with pytest.raises(RuntimeError):
        FederatedXGBoost().predict_proba(test_set)
    with pytest.raises(RuntimeError):
        FedAvgLogistic().predict_proba(test_set)
