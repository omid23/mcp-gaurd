"""OpenTelemetry setup and the trace context carried in MCP ``_meta``.

Span attributes follow the OpenTelemetry GenAI/MCP semantic conventions where
they exist (``mcp.method.name``, ``gen_ai.tool.name``,
``gen_ai.operation.name``); gateway-specific attributes use ``mcp_guard.*``.
The conventions are still marked Development upstream, so names may shift.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.propagate import extract

from .config import TelemetryConfig

log = logging.getLogger("mcp_guard")
_configured = False


def setup(cfg: TelemetryConfig) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    if not (cfg.otlp_endpoint or cfg.console):
        return  # API stays a no-op; zero overhead.
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": cfg.service_name}))
    if cfg.console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    if cfg.otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            log.warning("otlp_endpoint set but the 'otlp' extra is not installed; traces not exported")
        else:
            endpoint = cfg.otlp_endpoint.rstrip("/") + "/v1/traces"
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)


tracer = trace.get_tracer("mcp_guard")
_meter = metrics.get_meter("mcp_guard")
decisions_counter = _meter.create_counter(
    "mcp_guard.decisions", description="Guard decisions by type, source and outcome"
)
decision_latency = _meter.create_histogram(
    "mcp_guard.decision.duration", unit="ms", description="Time spent deciding, including model calls"
)


def context_from_meta(meta: Mapping[str, Any], headers: Mapping[str, str]):
    """Continue the caller's trace: W3C trace context may arrive in _meta (SEP-414) or headers."""
    carrier = {k: v for k, v in headers.items() if k.lower() in ("traceparent", "tracestate", "baggage")}
    for key in ("traceparent", "tracestate", "baggage"):
        if isinstance(meta.get(key), str):
            carrier[key] = meta[key]
    return extract(carrier)


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None
