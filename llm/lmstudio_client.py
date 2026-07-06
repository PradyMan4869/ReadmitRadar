"""
LM Studio client — local, OpenAI-compatible; PHI never leaves the machine.

Design rule: the LLM layer must never break the pipeline. If LM Studio is
down (or the openai package is missing), calls return None and callers fall
back to deterministic templates, clearly marked as such.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from config import LMSTUDIO
from observability.tracing import get_tracer

logger = logging.getLogger(__name__)


@dataclass
class Completion:
    """An LLM completion plus the Langfuse trace it was recorded under, so
    callers can attach an LLM-as-judge score to that same trace after the
    fact. `text` is None on failure (caller falls back to a template)."""
    text: Optional[str]
    trace_id: Optional[str] = None


class LMStudioClient:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or LMSTUDIO.base_url
        self.model = model or LMSTUDIO.model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            # api_key is required by the SDK but ignored by LM Studio
            self._client = OpenAI(base_url=self.base_url, api_key="lm-studio",
                                  timeout=LMSTUDIO.timeout_s)
        return self._client

    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(self.base_url.rstrip("/") + "/models", timeout=3)
            return r.ok
        except Exception:
            return False

    def complete(
        self,
        system: str,
        user: str,
        trace_name: str = "lmstudio-completion",
        temperature: float = 0.3,
        max_tokens: int = 400,
    ) -> Completion:
        """One chat completion. `.text` is None on any failure (caller
        falls back); the attempt and outcome are traced either way."""
        tracer = get_tracer()
        with tracer.trace(trace_name, model=self.model,
                          input={"system": system, "user": user}) as span:
            try:
                # Mistral-family chat templates reject the system role;
                # fold it into the user turn for broad model compatibility
                response = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": f"{system}\n\n{user}"},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content.strip()
                span.set_output(text, usage=getattr(response, "usage", None))
                return Completion(text=text, trace_id=span.trace_id)
            except Exception as e:
                logger.warning(f"LM Studio unavailable ({e}); using fallback")
                span.set_error(str(e))
                return Completion(text=None, trace_id=span.trace_id)
