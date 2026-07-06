"""SHAP explanations for individual readmission predictions."""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FEATURE_LABELS, FEATURE_UNITS
from .model import ReadmissionModel


@dataclass(frozen=True)
class Contribution:
    feature: str
    label: str
    value: float          # the patient's actual value
    unit: str
    shap: float           # signed log-odds contribution

    @property
    def direction(self) -> str:
        return "raises" if self.shap > 0 else "lowers"


class RiskExplainer:
    """Wraps shap.TreeExplainer for a fitted ReadmissionModel."""

    def __init__(self, model: ReadmissionModel):
        import shap
        model._require_fitted()
        self.model = model
        self.explainer = shap.TreeExplainer(model.clf)

    def explain_row(self, row: dict, top_k: int = 6) -> List[Contribution]:
        """Top-k features by |SHAP| for a single feature-row dict."""
        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        shap_values = np.asarray(self.explainer.shap_values(X))[0]
        order = np.argsort(-np.abs(shap_values))[:top_k]
        return [
            Contribution(
                feature=FEATURE_COLUMNS[i],
                label=FEATURE_LABELS[FEATURE_COLUMNS[i]],
                value=float(X.iloc[0, i]),
                unit=FEATURE_UNITS[FEATURE_COLUMNS[i]],
                shap=float(shap_values[i]),
            )
            for i in order
        ]

    def explain_frame(self, X: pd.DataFrame) -> np.ndarray:
        """Raw SHAP matrix for a batch (used by the UI waterfall)."""
        return np.asarray(self.explainer.shap_values(X[FEATURE_COLUMNS]))


def contributions_to_text(contribs: List[Contribution]) -> str:
    """Compact deterministic rendering, used as LLM input and as the
    offline fallback when LM Studio is unavailable."""
    lines = []
    for c in contribs:
        value = f"{c.value:g} {c.unit}".strip() if c.unit != "0/1" else (
            "yes" if c.value == 1 else "no")
        lines.append(f"- {c.label}: {value} ({c.direction} risk, "
                     f"SHAP {c.shap:+.3f})")
    return "\n".join(lines)
