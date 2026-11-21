"""
Federated training simulation across two hospitals.

Regimes, evaluated on held-out data:

  local        — one model per hospital, sees only its own data
                 (what each hospital can do alone)
  centralized  — one model on pooled data
                 (upper bound; requires moving PHI between hospitals,
                  which HIPAA makes a non-starter — included as baseline)
  federated    — no raw patient record ever leaves either hospital:
                   * FedAvg (true weight averaging): a logistic model whose
                     coefficient vector is averaged across clients over
                     multiple local-epoch rounds — the textbook algorithm
                     (McMahan et al., 2017). Only valid for models with a
                     dense weight vector, which is why it's the logistic
                     regime, not the tree regime, below.
                   * Federated gradient-boosted trees have no weight vector
                     to average. Instead, FederatedHistogramXGBoost grows
                     one shared tree ensemble by having each hospital send
                     only per-feature-bin gradient/hessian histograms (small
                     aggregate tables, never patient rows) to a coordinator,
                     which sums them to pick each split — mathematically
                     identical to training on pooled data. Leaf *values* are
                     then computed separately per hospital (personalized),
                     since each site's baseline risk and case mix differ —
                     so the deployed artifact is one shared tree structure
                     with two parallel sets of leaf outputs, not one
                     model-for-everyone.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TARGET
from .model import ReadmissionModel, bootstrap_ci, evaluate


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


# ── Histogram-aggregation federation for gradient-boosted trees ────────────
#
# Real federated XGBoost (as implemented by systems like NVFlare/FATE): at
# every tree split, each client reduces its local rows to a per-feature,
# per-bin histogram of (gradient_sum, hessian_sum, count) — a small table of
# aggregate numbers, not patient rows. A coordinator sums the per-client
# histograms and picks the split with the best XGBoost regularized gain.
# Only histograms and the resulting split choice cross the wire.

N_BINS = 256
DEFAULT_LAMBDA = 1.0   # L2 leaf-weight regularization (matches reg_lambda)
DEFAULT_GAMMA = 0.0    # minimum gain required to make a split


def _quantile_bin_edges(client_dfs: List[pd.DataFrame], n_bins: int = N_BINS) -> Dict[str, np.ndarray]:
    """
    Per-feature bin edges shared by every hospital and every tree/split —
    XGBoost's 'hist' method. Each hospital's own quantiles are pooled
    (sample-size-weighted) so the shared edges reflect the federation's
    overall distribution without any hospital sending raw rows to compute
    them; only these numeric edge values cross the wire.
    """
    edges: Dict[str, np.ndarray] = {}
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    for col in FEATURE_COLUMNS:
        local_quantiles = [df[col].quantile(quantiles).values for df in client_dfs]
        sizes = np.array([len(df) for df in client_dfs], dtype=float)
        merged = np.average(local_quantiles, axis=0, weights=sizes)
        edges[col] = np.unique(merged)
    return edges


@dataclass
class _Histogram:
    """Per-feature, per-bin (grad_sum, hess_sum, count) for one client's rows."""
    grad: Dict[str, np.ndarray]
    hess: Dict[str, np.ndarray]
    count: Dict[str, np.ndarray]

    @classmethod
    def build(cls, X_binned: Dict[str, np.ndarray], grad: np.ndarray,
              hess: np.ndarray, row_mask: np.ndarray, n_bins: int) -> "_Histogram":
        g, h, c = {}, {}, {}
        rows = np.nonzero(row_mask)[0]
        for col, bins in X_binned.items():
            b = bins[rows]
            g[col] = np.bincount(b, weights=grad[rows], minlength=n_bins)
            h[col] = np.bincount(b, weights=hess[rows], minlength=n_bins)
            c[col] = np.bincount(b, minlength=n_bins).astype(float)
        return cls(grad=g, hess=h, count=c)

    @staticmethod
    def sum_across_clients(histograms: List["_Histogram"]) -> "_Histogram":
        cols = histograms[0].grad.keys()
        g = {c: sum(h.grad[c] for h in histograms) for c in cols}
        h_ = {c: sum(h.hess[c] for h in histograms) for c in cols}
        cnt = {c: sum(h.count[c] for h in histograms) for c in cols}
        return _Histogram(grad=g, hess=h_, count=cnt)


def _leaf_value(grad_sum: float, hess_sum: float, lam: float) -> float:
    return -grad_sum / (hess_sum + lam)


def _split_gain(g_l: float, h_l: float, g_r: float, h_r: float, lam: float) -> float:
    g, h = g_l + g_r, h_l + h_r
    return 0.5 * (g_l ** 2 / (h_l + lam) + g_r ** 2 / (h_r + lam) - g ** 2 / (h + lam))


@dataclass
class _Node:
    is_leaf: bool = True
    feature: Optional[str] = None
    bin_threshold: Optional[int] = None   # split: bin_index <= threshold -> left
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    # leaf_values[client_index] = that hospital's personalized leaf output
    leaf_values: Optional[np.ndarray] = None


class _FederatedTree:
    """One boosted tree grown from cross-hospital-summed histograms, with
    per-hospital-personalized leaf values."""

    def __init__(self, max_depth: int, lam: float, gamma: float, min_child_weight: float):
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.root: Optional[_Node] = None

    def fit(self, X_binned: List[Dict[str, np.ndarray]], grads: List[np.ndarray],
            hess: List[np.ndarray], n_bins: int) -> "_FederatedTree":
        n_clients = len(X_binned)
        masks = [np.ones(len(g), dtype=bool) for g in grads]
        self.root = self._grow(X_binned, grads, hess, masks, n_bins, depth=0)
        return self

    def _grow(self, X_binned, grads, hess, masks, n_bins, depth) -> _Node:
        histograms = [
            _Histogram.build(X_binned[i], grads[i], hess[i], masks[i], n_bins)
            for i in range(len(X_binned))
        ]
        total = _Histogram.sum_across_clients(histograms)

        best = self._best_split(total, n_bins)
        if depth >= self.max_depth or best is None:
            return self._make_leaf(histograms)

        feature, threshold, gain = best
        if gain <= self.gamma:
            return self._make_leaf(histograms)

        left_masks, right_masks = [], []
        for i in range(len(X_binned)):
            bins = X_binned[i][feature]
            go_left = masks[i] & (bins <= threshold)
            go_right = masks[i] & ~go_left
            left_masks.append(go_left)
            right_masks.append(go_right)

        node = _Node(is_leaf=False, feature=feature, bin_threshold=threshold)
        node.left = self._grow(X_binned, grads, hess, left_masks, n_bins, depth + 1)
        node.right = self._grow(X_binned, grads, hess, right_masks, n_bins, depth + 1)
        return node

    def _best_split(self, total: _Histogram, n_bins: int):
        # Vectorized across all bins (and features) at once — the per-bin
        # Python loop this replaced was the dominant cost of tree growth.
        best_gain, best_feature, best_threshold = 0.0, None, None
        for col in total.grad:
            g_cum = np.cumsum(total.grad[col])[:-1]
            h_cum = np.cumsum(total.hess[col])[:-1]
            c_cum = np.cumsum(total.count[col])[:-1]
            g_total, h_total, c_total = total.grad[col].sum(), total.hess[col].sum(), total.count[col].sum()

            h_l, h_r = h_cum, h_total - h_cum
            c_l, c_r = c_cum, c_total - c_cum
            g_l, g_r = g_cum, g_total - g_cum

            valid = (c_l >= 1) & (c_r >= 1) & (h_l >= self.min_child_weight) & (h_r >= self.min_child_weight)
            if not valid.any():
                continue

            gains = 0.5 * (g_l ** 2 / (h_l + self.lam) + g_r ** 2 / (h_r + self.lam)
                            - g_total ** 2 / (h_total + self.lam))
            gains = np.where(valid, gains, -np.inf)

            b = int(np.argmax(gains))
            if gains[b] > best_gain:
                best_gain, best_feature, best_threshold = float(gains[b]), col, b
        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain

    def _make_leaf(self, histograms: List[_Histogram]) -> _Node:
        any_col = next(iter(histograms[0].grad))
        values = np.array([
            _leaf_value(h.grad[any_col].sum(), h.hess[any_col].sum(), self.lam)
            for h in histograms
        ])
        return _Node(is_leaf=True, leaf_values=values)

    def predict_margin(self, X_binned: Dict[str, np.ndarray], n_rows: int, client_index: int) -> np.ndarray:
        out = np.zeros(n_rows)
        for i in range(n_rows):
            node = self.root
            while not node.is_leaf:
                bins = X_binned[node.feature]
                node = node.left if bins[i] <= node.bin_threshold else node.right
            out[i] = node.leaf_values[client_index]
        return out


class FederatedHistogramXGBoost:
    """
    Federated gradient-boosted trees via histogram aggregation.

    Splits are chosen jointly from cross-hospital-summed gradient/hessian
    histograms — the resulting tree *structure* is what pooled-data
    training would produce, with no patient row ever leaving its hospital.
    Leaf values are computed per-hospital from that hospital's own
    gradient/hessian sums, so each site's deployed model shares the same
    learned splits but keeps predictions calibrated to its own population
    (different baseline risk / case mix). There is therefore no single
    "federated" prediction — call predict_proba(X, client_index=...) for
    the hospital whose leaves should be used.
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 4,
                 learning_rate: float = 0.1, reg_lambda: float = DEFAULT_LAMBDA,
                 gamma: float = DEFAULT_GAMMA, min_child_weight: float = 4.0,
                 n_bins: int = N_BINS):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lam = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.n_bins = n_bins
        self.bin_edges: Optional[Dict[str, np.ndarray]] = None
        self.trees: List[_FederatedTree] = []
        self.n_clients: int = 0
        self.base_score: float = 0.0

    def fit(self, client_dfs: List[pd.DataFrame]) -> "FederatedHistogramXGBoost":
        self.n_clients = len(client_dfs)
        self.bin_edges = _quantile_bin_edges(client_dfs, self.n_bins)
        # Every feature is clipped into [0, n_bins_actual - 1] by _bin(), so
        # one shared bin count works for every column's histogram array.
        self._n_bins_actual = max(len(edges) for edges in self.bin_edges.values())

        X_binned = [self._bin(df) for df in client_dfs]
        ys = [df[TARGET].values.astype(float) for df in client_dfs]

        pooled_y = np.concatenate(ys)
        pooled_mean = min(max(pooled_y.mean(), 1e-6), 1 - 1e-6)
        self.base_score = float(np.log(pooled_mean / (1 - pooled_mean)))

        # scale_pos_weight, same imbalance-handling ReadmissionModel uses —
        # only the pooled positive/negative *counts* (two numbers) are
        # exchanged to compute this, never patient rows, and both hospitals
        # apply the identical weight, so split selection stays equivalent
        # to pooled-data training.
        pos = max(int(pooled_y.sum()), 1)
        neg = len(pooled_y) - pos
        self.scale_pos_weight = neg / pos

        margins = [np.full(len(y), self.base_score) for y in ys]

        for _ in range(self.n_estimators):
            sample_weights = [
                np.where(y == 1, self.scale_pos_weight, 1.0) for y in ys
            ]
            grads = [
                w * (_sigmoid(m) - y) for m, y, w in zip(margins, ys, sample_weights)
            ]
            hess = [
                w * _sigmoid(m) * (1 - _sigmoid(m))
                for m, w in zip(margins, sample_weights)
            ]

            tree = _FederatedTree(self.max_depth, self.lam, self.gamma,
                                   self.min_child_weight)
            tree.fit(X_binned, grads, hess, self._n_bins_actual)
            self.trees.append(tree)

            for ci in range(self.n_clients):
                update = tree.predict_margin(X_binned[ci], len(margins[ci]), ci)
                margins[ci] = margins[ci] + self.learning_rate * update

        return self

    def _bin(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            col: np.clip(np.searchsorted(self.bin_edges[col], df[col].values, side="right") - 1,
                         0, len(self.bin_edges[col]) - 1)
            for col in FEATURE_COLUMNS
        }

    def margin(self, X: pd.DataFrame, client_index: int) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("FederatedHistogramXGBoost not fitted")
        X_binned = self._bin(X)
        margin = np.full(len(X), self.base_score)
        for tree in self.trees:
            margin = margin + self.learning_rate * tree.predict_margin(X_binned, len(X), client_index)
        return margin

    def predict_proba(self, X: pd.DataFrame, client_index: int) -> np.ndarray:
        return _sigmoid(self.margin(X, client_index))


# ── Experiment runner ────────────────────────────────────────────────────────

def run_comparison(
    hospital_dfs: Dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    fedavg_rounds: int = 10,
    xgb_params: dict | None = None,
    fed_hist_params: dict | None = None,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Train all regimes and evaluate.

    Most regimes are scored on the pooled held-out test set; the
    federated-histogram regime has personalized leaves (no single
    model-for-everyone, see FederatedHistogramXGBoost), so each hospital's
    version is scored only on that hospital's own test rows — the fair
    comparison is against that hospital's *local* model on the same rows,
    not against the pooled number other regimes use.

    Also includes each local model evaluated on the *other* hospital's
    share of the test set — the cross-hospital generalisation gap
    federated learning closes — and a bootstrap 95% CI on ROC-AUC/PR-AUC
    for every regime, since single-run differences of a few thousandths
    on ~3k test rows are well within sampling noise and should not be
    read as one regime "beating" another.
    """
    y_test = test_df[TARGET].values
    results: dict = {}

    def _record(key: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        m = evaluate(y_true, y_prob)
        m.update(bootstrap_ci(y_true, y_prob, n_resamples=n_bootstrap))
        results[key] = m

    # Local models
    local_models = {}
    for name, df in hospital_dfs.items():
        model = ReadmissionModel(xgb_params).fit(df)
        local_models[name] = model
        _record(f"local_{name}", y_test, model.predict_proba(test_df))

    # Cross-hospital: model from A scored only on B-origin test rows, and
    # vice versa — the honest measure of single-site generalisation failure,
    # and the real motivation for federated learning in this project.
    has_site_col = "hospital" in test_df.columns
    if has_site_col:
        for name, model in local_models.items():
            other = test_df[test_df["hospital"] != name]
            if len(other):
                _record(f"local_{name}_on_other_site",
                        other[TARGET].values, model.predict_proba(other))

    # Centralized (privacy-violating upper bound)
    pooled = pd.concat(hospital_dfs.values(), ignore_index=True)
    central = ReadmissionModel(xgb_params).fit(pooled)
    _record("centralized", y_test, central.predict_proba(test_df))

    # Federated histogram-aggregation XGBoost: one shared tree structure,
    # personalized leaves — evaluate each hospital's version on its own
    # site's test rows only.
    client_names = list(hospital_dfs.keys())
    fed_hist = FederatedHistogramXGBoost(**(fed_hist_params or {})).fit(
        list(hospital_dfs.values())
    )
    for ci, name in enumerate(client_names):
        rows = test_df[test_df["hospital"] == name] if has_site_col else test_df
        if len(rows):
            _record(f"federated_personalized_{name}",
                    rows[TARGET].values, fed_hist.predict_proba(rows, ci))

    # True FedAvg (logistic)
    fedavg = FedAvgLogistic(rounds=fedavg_rounds).fit(list(hospital_dfs.values()))
    _record("fedavg_logistic", y_test, fedavg.predict_proba(test_df))

    return results
