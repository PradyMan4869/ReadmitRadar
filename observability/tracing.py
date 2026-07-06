"""
Langfuse tracing wrapper (self-hosted, on-prem).

When Langfuse keys are unset or the package is missing, get_tracer()
returns a no-op tracer with the same interface — observability is
opt-in and can never take the pipeline down.
"""
import logging
from contextlib import contextmanager

from config import LANGFUSE

logger = logging.getLogger(__name__)


class _NoOpSpan:
    trace_id = None

    def set_output(self, output, usage=None):
        pass

    def set_error(self, message):
        pass


class _NoOpTracer:
    enabled = False

    @contextmanager
    def trace(self, name, **kwargs):
        yield _NoOpSpan()

    def score(self, trace_id, name, value, comment=None):
        pass

    def flush(self):
        pass


class _LangfuseSpan:
    def __init__(self, observation, trace_id):
        self._observation = observation
        self.trace_id = trace_id

    def set_output(self, output, usage=None):
        kwargs = {"output": output}
        if usage is not None:
            kwargs["usage_details"] = {
                "input": getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
            }
        self._observation.update(**kwargs)

    def set_error(self, message):
        self._observation.update(output=None, level="ERROR", status_message=message)


class _LangfuseTracer:
    enabled = True

    def __init__(self, client):
        self._client = client

    @contextmanager
    def trace(self, name, model=None, input=None):
        with self._client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=input
        ) as observation:
            trace_id = self._client.get_current_trace_id()
            yield _LangfuseSpan(observation, trace_id)

    def score(self, trace_id, name, value, comment=None):
        """Attach a numeric score to a past trace — used for LLM-as-judge
        ratings, which run after the generation they're judging has
        already completed and closed."""
        if trace_id is None:
            return
        try:
            self._client.create_score(
                name=name, value=value, data_type="NUMERIC",
                trace_id=trace_id, comment=comment,
            )
        except Exception as e:
            logger.warning(f"Langfuse score failed ({e})")

    def flush(self):
        self._client.flush()


_tracer = None


def get_tracer():
    """Singleton tracer: Langfuse when configured, no-op otherwise."""
    global _tracer
    if _tracer is not None:
        return _tracer

    if LANGFUSE.enabled:
        try:
            from langfuse import Langfuse
            client = Langfuse(
                host=LANGFUSE.host,
                public_key=LANGFUSE.public_key,
                secret_key=LANGFUSE.secret_key,
            )
            _tracer = _LangfuseTracer(client)
            logger.info(f"Langfuse tracing enabled → {LANGFUSE.host}")
            return _tracer
        except Exception as e:
            logger.warning(f"Langfuse init failed ({e}); tracing disabled")

    _tracer = _NoOpTracer()
    return _tracer
