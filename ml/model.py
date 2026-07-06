"""XGBoost readmission classifier wrapper."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS, TARGET

DEFAULT_PARAMS = dict(
    max_depth=4,
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=4,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)


class ReadmissionModel:
    """Thin wrapper enforcing the feature schema around XGBClassifier."""

    def __init__(self, params: Optional[dict] = None, calibrate: bool = False):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.clf: Optional[xgb.XGBClassifier] = None
        self.calibrate = calibrate
        self._calibrator: Optional[LogisticRegression] = None

    def fit(self, df: pd.DataFrame) -> "ReadmissionModel":
        X = df[FEATURE_COLUMNS]
        y = df[TARGET]

        if self.calibrate:
            # scale_pos_weight distorts predict_proba into a ranking score,
            # not a true probability (mean output drifts well above the
            # actual base rate). Fit XGBoost on a train split, then fit a
            # 1-D Platt (sigmoid) calibrator mapping its raw margin -> a
            # probability that matches the held-out calibration split, so
            # predict_proba is both well-ranked (AUC) and well-calibrated
            # (mean predicted ~= actual base rate).
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X, y, test_size=0.2, random_state=self.params["random_state"],
                stratify=y,
            )
            pos = max(int(y_fit.sum()), 1)
            neg = len(y_fit) - pos
            self.clf = xgb.XGBClassifier(**self.params, scale_pos_weight=neg / pos)
            self.clf.fit(X_fit, y_fit)
            margin_cal = self.clf.predict(X_cal, output_margin=True)
            self._calibrator = LogisticRegression()
            self._calibrator.fit(margin_cal.reshape(-1, 1), y_cal)
        else:
            # scale_pos_weight keeps the minority (readmitted) class visible
            pos = max(int(y.sum()), 1)
            neg = len(y) - pos
            self.clf = xgb.XGBClassifier(
                **self.params, scale_pos_weight=neg / pos
            )
            self.clf.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        if self._calibrator is not None:
            margin = self.clf.predict(X[FEATURE_COLUMNS], output_margin=True)
            return self._calibrator.predict_proba(margin.reshape(-1, 1))[:, 1]
        return self.clf.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def predict_row(self, row: dict) -> float:
        """Risk probability for a single feature-row dict."""
        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        return float(self.predict_proba(X)[0])

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        """Raw log-odds output — used by federated margin averaging."""
        self._require_fitted()
        return self.clf.predict(X[FEATURE_COLUMNS], output_margin=True)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        self.clf.save_model(str(path))
        if self._calibrator is not None:
            cal_path = Path(path).with_suffix(".calibrator.json")
            cal_path.write_text(json.dumps({
                "coef": self._calibrator.coef_.tolist(),
                "intercept": self._calibrator.intercept_.tolist(),
                "classes": self._calibrator.classes_.tolist(),
            }), encoding="utf-8")

    def load(self, path: str | Path) -> "ReadmissionModel":
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(str(path))
        cal_path = Path(path).with_suffix(".calibrator.json")
        if cal_path.exists():
            data = json.loads(cal_path.read_text(encoding="utf-8"))
            cal = LogisticRegression()
            cal.coef_ = np.array(data["coef"])
            cal.intercept_ = np.array(data["intercept"])
            cal.classes_ = np.array(data["classes"])
            self._calibrator = cal
        return self

    def _require_fitted(self) -> None:
        if self.clf is None:
            raise RuntimeError("model not fitted/loaded")


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Standard binary-classification metrics used across all regimes."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
        precision_score, recall_score, roc_auc_score,
    )
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
