# Agent Harness
from __future__ import annotations
import hashlib
import json
import logging
import threading
import time
import traceback
import uuid

from collections import Counter
from dataclasses import asdict,dataclass,field
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Optional
from langchain_core.callbacks import BaseCallbackHandler

# 1. EXCEPTIONS

class HarnessError(Exception):
    """Base HArness exception"""

class LoopDetectionError(HarnessError):
    """Raised when repeated agent behavior is detected"""

class ExecutionTimeoutError(HarnessError):
    """Raised when total execution time exceeds the limit"""

class TokenBudgetExceededError():
    """Raised when LLM usage exceeds the request token budget"""

class ToolBudgetExceededError():
    """Raised when the number of tool exceeds the limit."""


# 2. Utility Functions

def utc_now()-> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value))
        except Exception:
            pass

    return str(value)


def state_fingerprint(state: dict[str, Any]) -> str:
    """
    Creates a behavioral fingerprint.

    Iteration is deliberately excluded because otherwise identical
    loops would produce different fingerprints.
    """

    important_state = {
        "selected_restaurant": state.get("selected_restaurant"),
        "attempted_restaurants": state.get(
            "attempted_restaurants",
            [],
        ),
        "plan": state.get("plan"),
        "tool_result": state.get("tool_result"),
        "goal_achieved": state.get("goal_achieved"),
        "evaluation_message": state.get(
            "evaluation_message"
        ),
    }

    serialized = json.dumps(
        json_safe(important_state),
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

# 3. Metric Modul

@dataclass
class TokenMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class LLMCallMetric:
    call_id: str
    model_name: Optional[str]
    started_at: str
    completed_at: Optional[str] = None
    latency_ms: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    success: bool = False
    error: Optional[str] = None


@dataclass
class ToolCallMetric:
    tool_name: str
    restaurant: Optional[str]
    started_at: str
    completed_at: str
    latency_ms: float
    technically_successful: bool
    business_result: str
    output: dict[str, Any]
    error: Optional[str] = None


@dataclass
class NodeMetric:
    sequence: int
    node_name: str
    completed_at: str
    elapsed_since_previous_node_ms: float
    updated_fields: list[str]


@dataclass
class EvaluationResult:
    evaluator: str
    score: float
    passed: bool
    reasons: list[str]
    factual_consistency: bool
    goal_satisfied: bool
    evaluated_at: str


@dataclass
class RequestMetrics:
    request_id: str
    thread_id: str
    agent_name: str
    started_at: str

    completed_at: Optional[str] = None
    status: str = "Running"
    duration_ms: float = 0.0

    iteration_count: int = 0
    graph_event_count: int = 0
    node_execution_count: int = 0
    tool_call_count: int = 0
    llm_call_count: int = 0

    loop_detected: bool = False
    loop_reason: Optional[str] = None

    fallback_used: bool = False
    fallback_count: int = 0

    tokens: TokenMetrics = field(
        default_factory=TokenMetrics
    )
    llm_calls: list[LLMCallMetric] = field(
        default_factory=list
    )

    tool_calls: list[ToolCallMetric] = field(
        default_factory=list
    )

    nodes: list[NodeMetric] = field(
        default_factory=list
    )

    evaluation: Optional[EvaluationResult] = None

    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


@dataclass
class HarnessResult:
    succeeded: bool
    request_id: str
    final_state: dict[str, Any]
    metrics: RequestMetrics


# ============================================================
# 4. AUDIT LOGGER
# ============================================================

class AuditLogger:
    """
    Thread-safe append-only JSONL audit logger.

    Production alternatives:
    - Azure Application Insights
    - Azure Monitor
    - Log Analytics
    - Event Hub
    - PostgreSQL
    """

    SENSITIVE_KEYS = {
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
    }

    def __init__(
        self,
        path: str,
        record_content: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.record_content = record_content
        self._lock = threading.Lock()

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            redacted = {}

            for key, item in value.items():
                lower_key = str(key).lower()

                if any(
                    sensitive in lower_key
                    for sensitive in self.SENSITIVE_KEYS
                ):
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = self.redact(item)

            return redacted

        if isinstance(value, list):
            return [self.redact(item) for item in value]

        return value
    def log(
        self,
        event_type: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> None:
        safe_payload = self.redact(
            json_safe(payload)
        )

        record = {
            "timestamp": utc_now(),
            "event_type": event_type,
            "request_id": request_id,
            "payload": safe_payload,
        }

        line = json.dumps(
            record,
            ensure_ascii=False,
            default=str,
        )

        with self._lock:
            with self.path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(line + "\n")


# ============================================================
# 5. LLM CALLBACK
# ============================================================

class HarnessLLMCallback(BaseCallbackHandler):
    """
    Captures:
    - LLM call count
    - model latency
    - token usage
    - errors
    - model/deployment information

    It intentionally does not store prompts or generated text.
    """

    def __init__(
        self,
        metrics: RequestMetrics,
        audit_logger: AuditLogger,
        max_total_tokens: int,
    ) -> None:
        self.metrics = metrics
        self.audit_logger = audit_logger
        self.max_total_tokens = max_total_tokens

        self._active_calls: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:

        call_id = str(run_id)

        model_name = (
            serialized.get("kwargs", {}).get(
                "azure_deployment"
            )
            or serialized.get("kwargs", {}).get(
                "model_name"
            )
            or serialized.get("name")
        )

        with self._lock:
            # Remaining code is cut off in the image.
            self._active_calls[call_id] = {
                "started_perf": time.perf_counter(),
                "metric": LLMCallMetric(
                    call_id=call_id,
                    model_name=model_name,
                    started_at=utc_now(),
                ),
            }

        self.audit_logger.write(
            "llm_call_started",
            self.metrics.request_id,
            {
                "call_id": call_id,
                "model_name": model_name,
                "parent_run_id": (
                    str(parent_run_id)
                    if parent_run_id
                    else None
                ),
                "message_count": sum(
                    len(batch)
                    for batch in messages
                ),
                "tags": tags or [],
                "metadata": metadata or {},
            },
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        call_id = str(run_id)

        with self._lock:
            active = self._active_calls.pop(
                call_id,
                None,
            )

        if active is None:
            return

        metric: LLMCallMetric = active["metric"]

        metric.completed_at = utc_now()

        metric.latency_ms = round(
            (
                time.perf_counter()
                - active["started_perf"]
            )
            * 1000,
            3,
        )

        usage = self._extract_usage(response)

        metric.input_tokens = usage["input_tokens"]
        metric.output_tokens = usage["output_tokens"]
        metric.total_tokens = usage["total_tokens"]
        metric.reasoning_tokens = usage[
            "reasoning_tokens"
        ]
        metric.cached_tokens = usage["cached_tokens"]

        metric.success = True

        with self._lock:
            self.metrics.llm_calls.append(metric)
            self.metrics.llm_call_count += 1

            self.metrics.tokens.input_tokens += (
                metric.input_tokens
            )

            self.metrics.tokens.output_tokens += (
                metric.output_tokens
            )
            self.metrics.tokens.total_tokens += (
                metric.total_tokens
            )

            self.metrics.tokens.reasoning_tokens += (
                metric.reasoning_tokens
            )

            self.metrics.tokens.cached_tokens += (
                metric.cached_tokens
            )

        self.audit_logger.write(
            "llm_call_completed",
            self.metrics.request_id,
            asdict(metric),
        )

        if (
            self.metrics.tokens.total_tokens
            > self.max_total_tokens
        ):
            raise TokenBudgetExceededError(
                "Token budget exceeded. "
                f"Used={self.metrics.tokens.total_tokens}, "
                f"Limit={self.max_total_tokens}."
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        call_id = str(run_id)

        with self._lock:
            active = self._active_calls.pop(
                call_id,
                None,
            )

        if active is None:
            return

        metric: LLMCallMetric = active["metric"]

        metric.completed_at = utc_now()

        metric.latency_ms = round(
            (
                time.perf_counter()
                - active["started_perf"]
            )
            * 1000,
            3,
        )

        metric.success = False
        metric.error = str(error)

        with self._lock:
            self.metrics.llm_calls.append(metric)
            self.metrics.llm_call_count += 1

        self.audit_logger.write(
            "llm_call_failed",
            self.metrics.request_id,
            asdict(metric),
        )

    @staticmethod
    def _extract_usage(
        response: Any,
    ) -> dict[str, int]:
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokes":0,
            "cached_tokens":0,
        }