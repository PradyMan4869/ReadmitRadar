"""
AutoGen two-agent deliberation: a Clinician and a Risk Analyst discuss a
prediction before the final summary is issued.

Primary path uses pyautogen with LM Studio as the (local) model backend.
When pyautogen is not installed or LM Studio is down, a manual two-turn
exchange through LMStudioClient (or its template fallback) produces the
same transcript structure, so the UI renders identically either way.
"""
import logging
from typing import List

from config import LMSTUDIO
from ml.explain import Contribution, contributions_to_text
from ml.features import RISK_BANDS
from observability.judge import judge_and_score
from observability.tracing import get_tracer

from .lmstudio_client import LMStudioClient

logger = logging.getLogger(__name__)

_CLINICIAN_SYSTEM = (
    "You are an attending physician reviewing a discharge. Assess whether "
    "the model's readmission risk estimate is clinically plausible given "
    "the cited drivers. Be concise (2-3 sentences); note anything the "
    "model may be over- or under-weighting."
)
_ANALYST_SYSTEM = (
    "You are a risk analyst who owns this prediction model. Respond to the "
    "clinician's assessment: explain what the SHAP drivers support, concede "
    "any fair critique, and give a final risk disposition with one concrete "
    "care-team action. Be concise (2-3 sentences)."
)


def _case_summary(risk: float, contributions: List[Contribution]) -> str:
    return (
        f"Case: predicted 30-day readmission risk {risk:.0%} "
        f"({RISK_BANDS.label(risk)}).\n"
        f"Top model drivers:\n{contributions_to_text(contributions)}"
    )


def deliberate(risk: float, contributions: List[Contribution]) -> dict:
    """
    Returns {"transcript": [{"role", "content"}...], "source"} where source
    is "autogen", "manual-llm", or "template".
    """
    case = _case_summary(risk, contributions)
    client = LMStudioClient()

    if client.is_available():
        try:
            return _autogen_deliberation(case)
        except ImportError:
            logger.info("pyautogen not installed; using manual two-turn loop")
        except Exception as e:
            logger.warning(f"AutoGen deliberation failed ({e}); manual loop")
        result = _manual_deliberation(case, client)
        if result:
            return result

    return _template_deliberation(risk, contributions)


def _autogen_deliberation(case: str) -> dict:
    import autogen

    llm_config = {"config_list": [{
        "model": LMSTUDIO.model,
        "base_url": LMSTUDIO.base_url,
        "api_key": "lm-studio",
        "price": [0.0, 0.0],  # local model — zero cost
    }]}
    clinician = autogen.AssistantAgent(
        "Clinician", system_message=_CLINICIAN_SYSTEM, llm_config=llm_config,
        max_consecutive_auto_reply=1,
    )
    analyst = autogen.AssistantAgent(
        "RiskAnalyst", system_message=_ANALYST_SYSTEM, llm_config=llm_config,
        max_consecutive_auto_reply=1,
    )

    with get_tracer().trace("autogen-deliberation", model=LMSTUDIO.model,
                            input=case) as span:
        result = analyst.initiate_chat(clinician, message=case, max_turns=2)
        transcript = [
            {"role": m.get("name", m.get("role", "agent")),
             "content": m.get("content", "")}
            for m in result.chat_history
        ]
        span.set_output(transcript)

    transcript_text = "\n".join(
        f"{t['role']}: {t['content']}" for t in transcript)
    judge_and_score(
        LMStudioClient(), case, transcript_text, span.trace_id,
        trace_name="judge-autogen-deliberation",
    )

    return {"transcript": transcript, "source": "autogen"}


def _manual_deliberation(case: str, client: LMStudioClient) -> dict | None:
    clinician = client.complete(_CLINICIAN_SYSTEM, case,
                                trace_name="deliberation-clinician")
    if not clinician.text:
        return None
    judge_and_score(
        client, case, clinician.text, clinician.trace_id,
        trace_name="judge-deliberation-clinician",
    )

    analyst = client.complete(
        _ANALYST_SYSTEM,
        f"{case}\n\nClinician's assessment:\n{clinician.text}",
        trace_name="deliberation-analyst",
    )
    if not analyst.text:
        return None
    judge_and_score(
        client, case, analyst.text, analyst.trace_id,
        trace_name="judge-deliberation-analyst",
    )

    return {
        "transcript": [
            {"role": "case", "content": case},
            {"role": "Clinician", "content": clinician.text},
            {"role": "RiskAnalyst", "content": analyst.text},
        ],
        "source": "manual-llm",
    }


def _template_deliberation(risk: float,
                           contributions: List[Contribution]) -> dict:
    band = RISK_BANDS.label(risk)
    raising = [c.label for c in contributions if c.shap > 0][:3]
    return {
        "transcript": [
            {"role": "case", "content": _case_summary(risk, contributions)},
            {"role": "Clinician",
             "content": f"The {band.lower()}-risk estimate is directionally "
                        f"consistent with {', '.join(raising).lower() or 'the cited drivers'}. "
                        "[Deterministic fallback — LM Studio offline.]"},
            {"role": "RiskAnalyst",
             "content": f"Disposition: treat as {band} risk; recommend care-team "
                        "review of the listed drivers before discharge. "
                        "[Deterministic fallback — LM Studio offline.]"},
        ],
        "source": "template",
    }
