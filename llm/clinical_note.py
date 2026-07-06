"""SHAP contributions → plain-English clinical rationale."""
from typing import List

from ml.explain import Contribution, contributions_to_text
from ml.features import RISK_BANDS
from observability.judge import judge_and_score

from .lmstudio_client import LMStudioClient

_SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant writing for discharge "
    "planners. Given a 30-day readmission risk score and the model's "
    "feature-level drivers (SHAP values), write a 3-5 sentence plain-English "
    "rationale. Name the concrete clinical facts (lab values, comorbidities, "
    "utilization history), state which raise vs lower the risk, and end with "
    "one actionable suggestion (e.g. schedule follow-up, medication "
    "reconciliation). Do not invent facts not in the input. Do not give a "
    "diagnosis. This is decision support, not medical advice."
)


def generate_clinical_note(
    risk: float,
    contributions: List[Contribution],
    client: LMStudioClient = None,
) -> dict:
    """
    Returns {"note", "source"} where source is "llm" or "template".
    The template path keeps the pipeline fully functional offline.
    """
    client = client or LMStudioClient()
    band = RISK_BANDS.label(risk)
    drivers = contributions_to_text(contributions)
    user_prompt = (
        f"30-day readmission risk: {risk:.0%} ({band}).\n"
        f"Model drivers (top SHAP contributions):\n{drivers}"
    )

    completion = client.complete(_SYSTEM_PROMPT, user_prompt,
                                 trace_name="clinical-note")
    if completion.text:
        judge_and_score(
            client, user_prompt, completion.text, completion.trace_id,
            trace_name="judge-clinical-note",
        )
        return {"note": completion.text, "source": "llm"}

    return {"note": _template_note(risk, band, contributions),
            "source": "template"}


def _template_note(risk: float, band: str,
                   contributions: List[Contribution]) -> str:
    raising = [c for c in contributions if c.shap > 0]
    lowering = [c for c in contributions if c.shap < 0]
    parts = [f"{band} risk of 30-day readmission ({risk:.0%})."]
    if raising:
        parts.append("Risk is driven by: "
                     + "; ".join(c.label.lower() for c in raising) + ".")
    if lowering:
        parts.append("Mitigating factors: "
                     + "; ".join(c.label.lower() for c in lowering) + ".")
    parts.append("[Deterministic fallback note — LM Studio offline.]")
    return " ".join(parts)
