"""
Federated training simulation across two hospitals.

Three regimes, evaluated on the same pooled held-out test set:

  local        — one model per hospital, sees only its own data
                 (what each hospital can do alone)
  centralized  — one model on pooled data
                 (upper bound; requires moving PHI between hospitals,
                  which HIPAA makes a non-starter — included as baseline)
  federated    — no raw records leave either hospital:
                   * FedAvg (true weight averaging): a logistic model whose
                     coefficient vector is averaged across clients over
                     multiple local-epoch rounds — the textbook algorithm.
                   * Federated XGBoost: gradient-boosted trees have no dense
                     weight vector to average, so the tree-adapted equivalent
                     aggregates per-hospital boosters by averaging their
                     raw margins (log-odds). Only model parameters cross
                     the wire in both cases, never patient rows.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TARGET
from .model import ReadmissionModel, evaluate


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ── True FedAvg on a logistic model ──────────────────────────────────────────

@dataclass
class FedAvgLogistic:
    """
    Federated Averaging (McMahan et al., 2017) on logistic regression.

    Each round: every client runs `local_epochs` of gradient descent from
    the shared global weights on its own data; the server averages the
    resulting weight vectors (weighted by client sample count).
    """
    rounds: int = 10
    local_epochs: int = 5
    lr: float = 0.1

    def __post_init__(self):
        self.weights: np.ndarray | None = None
        self.scaler: StandardScaler | None = None

    def fit(self, client_dfs: List[pd.DataFrame]) -> "FedAvgLogistic":
        # Scaling stats are computed per-client and averaged — a practical
        # federated preprocessing step (only aggregates cross the wire).
        self.scaler = self._federated_scaler(client_dfs)
        clients = [
            (self.scaler.transform(df[FEATURE_COLUMNS].values), df[TARGET].values)
            for df in client_dfs
        ]
        n_features = clients[0][0].shape[1]
        w = np.zeros(n_features + 1)  # + intercept
        sizes = np.array([len(y) for _, y in clients], dtype=float)

        for _ in range(self.rounds):
            client_weights = []
            for X, y in clients:
                w_local = w.copy()
                for _ in range(self.local_epochs):
                    z = X @ w_local[:-1] + w_local[-1]
                    err = _sigmoid(z) - y
                    grad_w = X.T @ err / len(y)
                    grad_b = err.mean()
                    w_local[:-1] -= self.lr * grad_w
                    w_local[-1] -= self.lr * grad_b
                client_weights.append(w_local)
            # FedAvg step: sample-size-weighted average of client weights
            w = np.average(client_weights, axis=0, weights=sizes)

        self.weights = w
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("FedAvgLogistic not fitted")
        Xs = self.scaler.transform(X[FEATURE_COLUMNS].values)
        return _sigmoid(Xs @ self.weights[:-1] + self.weights[-1])

    @staticmethod
    def _federated_scaler(client_dfs: List[pd.DataFrame]) -> StandardScaler:
        scalers = [StandardScaler().fit(df[FEATURE_COLUMNS].values)
                   for df in client_dfs]
        sizes = np.array([len(df) for df in client_dfs], dtype=float)
        merged = StandardScaler()
        merged.mean_ = np.average([s.mean_ for s in scalers], axis=0, weights=sizes)
        merged.var_ = np.average([s.var_ for s in scalers], axis=0, weights=sizes)
        scale = np.sqrt(merged.var_)
        # constant features (zero variance) pass through unscaled, as sklearn does
        merged.scale_ = np.where(scale == 0, 1.0, scale)
        merged.n_features_in_ = scalers[0].n_features_in_
        return merged


# ── Tree-adapted federation for XGBoost ─────────────────────────────────────

class FederatedXGBoost:
    """
    Aggregates independently trained per-hospital XGBoost models by
    averaging their raw margins (log-odds). Equivalent to a uniform
    ensemble in probability-calibrated space; no raw data is shared.
    """

    def __init__(self, params: dict | None = None):
        self.params = params
        self.members: List[ReadmissionModel] = []

    def fit(self, client_dfs: List[pd.DataFrame]) -> "FederatedXGBoost":
        self.members = [
            ReadmissionModel(self.params).fit(df) for df in client_dfs
        ]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.members:
            raise RuntimeError("FederatedXGBoost not fitted")
        margins = np.mean([m.margin(X) for m in self.members], axis=0)
        return _sigmoid(margins)


# ── Experiment runner ────────────────────────────────────────────────────────

def run_comparison(
    hospital_dfs: Dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    fedavg_rounds: int = 10,
    xgb_params: dict | None = None,
) -> dict:
    """
    Train all regimes and evaluate on the pooled test set.

    Returns {regime_name: metrics dict}; also includes each local model
    evaluated on the *other* hospital's share of the test set, which is
    the cross-hospital generalisation gap federated learning closes.
    """
    y_test = test_df[TARGET].values
    results: dict = {}

    # Local models
    local_models = {}
    for name, df in hospital_dfs.items():
        model = ReadmissionModel(xgb_params).fit(df)
        local_models[name] = model
        results[f"local_{name}"] = evaluate(y_test, model.predict_proba(test_df))

    # Cross-hospital: model from A scored only on B-origin test rows, and
    # vice versa — the honest measure of single-site generalisation failure
    if "hospital" in test_df.columns:
        for name, model in local_models.items():
            other = test_df[test_df["hospital"] != name]
            if len(other):
                results[f"local_{name}_on_other_site"] = evaluate(
                    other[TARGET].values, model.predict_proba(other)
                )

    # Centralized (privacy-violating upper bound)
    pooled = pd.concat(hospital_dfs.values(), ignore_index=True)
    central = ReadmissionModel(xgb_params).fit(pooled)
    results["centralized"] = evaluate(y_test, central.predict_proba(test_df))

    # Federated XGBoost (margin averaging)
    fed_xgb = FederatedXGBoost(xgb_params).fit(list(hospital_dfs.values()))
    results["federated_xgboost"] = evaluate(y_test, fed_xgb.predict_proba(test_df))

    # True FedAvg (logistic)
    fedavg = FedAvgLogistic(rounds=fedavg_rounds).fit(list(hospital_dfs.values()))
    results["fedavg_logistic"] = evaluate(y_test, fedavg.predict_proba(test_df))

    return results
