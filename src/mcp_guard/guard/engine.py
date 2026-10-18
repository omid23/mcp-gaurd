"""Combines deterministic policy with a decision model into allow / ask / deny."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import ServerConfig, Settings
from ..policy import STRONG_DESCRIPTION, STRONG_OUTPUT, PolicyEngine, description_signals, output_signals
from ..protocol import sha256_hex
from ..redact import redact
from .models import DecisionBackend, DecisionError, DecisionResponse, Question
from .questions import DESCRIPTION_QUESTIONS, OUTPUT_QUESTIONS, call_questions

Decision = Literal["allow", "ask", "deny"]


@dataclass
class Verdict:
    decision: Decision
    source: Literal["policy", "model", "fallback", "fail_mode", "static"]
    reasons: list[str] = field(default_factory=list)
    risks: dict[str, float] = field(default_factory=dict)
    signals: dict[str, list[str]] = field(default_factory=dict)
    backend: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    input_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    @property
    def max_risk(self) -> float:
        return max(self.risks.values(), default=0.0)

    def to_meta(self) -> dict:
        meta: dict[str, Any] = {"decision": self.decision, "source": self.source}
        if self.reasons:
            meta["reasons"] = self.reasons
        if self.risks:
            meta["risks"] = {k: round(v, 3) for k, v in self.risks.items()}
        if self.backend:
            meta["backend"] = self.backend
        return meta


class _TTLCache:
    def __init__(self, size: int, ttl_s: float):
        self.size, self.ttl_s = size, ttl_s
        self._data: OrderedDict[str, tuple[float, DecisionResponse]] = OrderedDict()

    def get(self, key: str) -> DecisionResponse | None:
        item = self._data.get(key)
        if item is None:
            return None
        if item[0] < time.monotonic():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return item[1]

    def put(self, key: str, value: DecisionResponse, ttl_s: float | None = None) -> None:
        self._data[key] = (time.monotonic() + (ttl_s or self.ttl_s), value)
        self._data.move_to_end(key)
        while len(self._data) > self.size:
            self._data.popitem(last=False)


def _truncate(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"…[+{len(value) - limit} chars]"
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value[:50]]
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in list(value.items())[:50]}
    return value


class GuardEngine:
    def __init__(
        self,
        settings: Settings,
        backend: DecisionBackend,
        fallback: DecisionBackend | None = None,
        policy: PolicyEngine | None = None,
        static_checks: bool = True,
    ):
        self.settings = settings
        # static_checks=False sends every description/output to the model, so a
        # backend can be measured on its own in evals.
        self.static_checks = static_checks
        self.backend = backend
        self.fallback = fallback
        self.policy = policy or PolicyEngine(settings.policy)
        dc = settings.decision
        self._cache = _TTLCache(dc.cache_size, dc.cache_ttl_s)

    # -- model plumbing ----------------------------------------------------------------

    def _prepare(self, state: dict) -> dict:
        limit = self.settings.decision.max_state_chars
        state = _truncate(state, limit)
        if self.settings.gateway.redact_before_model:
            state = redact(state)
        return state

    async def _ask_model(
        self, state: dict, questions: dict[str, Question], cache_ttl: float | None = None
    ) -> tuple[DecisionResponse | None, str, str | None]:
        """Returns (response, source, error). source is model|fallback|fail_mode."""
        key = sha256_hex({"b": self.backend.name, "s": state, "q": sorted(questions)})
        cached = self._cache.get(key)
        if cached is not None:
            return cached, "model", None
        try:
            resp = await self.backend.evaluate(state, questions)
            self._cache.put(key, resp, cache_ttl)
            return resp, "model", None
        except DecisionError as exc:
            error = str(exc)
        if self.fallback is not None and self.fallback is not self.backend:
            try:
                return await self.fallback.evaluate(state, questions), "fallback", error
            except DecisionError as exc:  # pragma: no cover - heuristic never raises
                error = f"{error}; fallback: {exc}"
        return None, "fail_mode", error

    def _classify(self, risks: dict[str, float], deny: float, ask: float) -> Decision:
        top = max(risks.values(), default=0.0)
        if top >= deny:
            return "deny"
        if top >= ask:
            return "ask"
        return "allow"

    # -- public checks -----------------------------------------------------------------

    async def check_call(
        self,
        server_id: str,
        server: ServerConfig,
        tool: dict | None,
        arguments: Any,
        intent: str | None = None,
    ) -> Verdict:
        started = time.perf_counter()
        tool = tool or {}
        name = str(tool.get("name", ""))
        pol = self.policy.evaluate_call(server_id, server, name, arguments, intent)
        if pol.action != "guard":
            return Verdict(
                pol.action, "policy", pol.reasons, signals=pol.signals,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        state: dict[str, Any] = {
            "kind": "tool_call",
            "server": server_id,
            "tool": {
                "name": name,
                "description": tool.get("description", ""),
                "annotations": tool.get("annotations") or {},
            },
            "arguments": arguments or {},
        }
        if pol.signals:
            state["deterministic_signals"] = pol.signals
        if intent:
            state["user_intent"] = intent
        questions = call_questions(with_intent=bool(intent))
        resp, source, error = await self._ask_model(self._prepare(state), questions)
        return self._verdict_from(resp, source, error, pol.signals, started, self.settings.decision.thresholds.deny,
                                  self.settings.decision.thresholds.ask, fail_closed=True)

    async def check_description(self, server_id: str, tool: dict, siblings: list[str]) -> Verdict:
        started = time.perf_counter()
        cfg = self.settings.description_scan
        text = " ".join(str(tool.get(k) or "") for k in ("title", "description"))
        schema_text = _schema_descriptions(tool.get("inputSchema"))
        if schema_text:
            text = f"{text}\n{schema_text}"
        signals = description_signals(text, [s for s in siblings if s != tool.get("name")])
        strong = [s for s in signals if s in STRONG_DESCRIPTION]
        if strong and self.static_checks:
            return Verdict(
                "deny", "static", [f"description contains {', '.join(strong)}"], signals=signals,
                risks={"hidden_instructions": 1.0}, latency_ms=(time.perf_counter() - started) * 1000,
            )
        state = {
            "kind": "tool_description",
            "server": server_id,
            "tool_name": tool.get("name", ""),
            "description": text,
            "other_tools": [s for s in siblings if s != tool.get("name")][:40],
        }
        # Descriptions only change when the fingerprint changes, so cache them for long.
        resp, source, error = await self._ask_model(self._prepare(state), DESCRIPTION_QUESTIONS, cache_ttl=3600)
        return self._verdict_from(resp, source, error, signals, started, cfg.threshold, cfg.threshold,
                                  fail_closed=False)

    async def check_output(self, server_id: str, tool_name: str, text: str) -> Verdict:
        started = time.perf_counter()
        cfg = self.settings.output_scan
        if not text.strip():
            return Verdict("allow", "static", latency_ms=0.0)
        text = text[: cfg.max_chars]
        signals = output_signals(text)
        if self.static_checks and any(s in STRONG_OUTPUT or s == "hidden_unicode" for s in signals):
            return Verdict(
                "deny", "static", [f"output contains {', '.join(sorted(signals))}"], signals=signals,
                risks={"injection": 1.0}, latency_ms=(time.perf_counter() - started) * 1000,
            )
        state = {"kind": "tool_output", "server": server_id, "tool_name": tool_name, "output": text}
        resp, source, error = await self._ask_model(self._prepare(state), OUTPUT_QUESTIONS)
        return self._verdict_from(resp, source, error, signals, started, cfg.threshold, cfg.threshold,
                                  fail_closed=False)

    def _verdict_from(
        self,
        resp: DecisionResponse | None,
        source: str,
        error: str | None,
        signals: dict,
        started: float,
        deny_t: float,
        ask_t: float,
        fail_closed: bool,
    ) -> Verdict:
        elapsed = (time.perf_counter() - started) * 1000
        if resp is None:
            # Calls fail closed (ask or deny per config); scans fail open with a reason logged,
            # because hiding every tool when the model is down would take the gateway offline.
            decision: Decision = self.settings.gateway.fail_mode if fail_closed else "allow"
            return Verdict(decision, "fail_mode", [f"decision model unavailable: {error}"], signals=signals,
                           latency_ms=elapsed)
        risks = {qid: a.p_yes for qid, a in resp.answers.items()}
        decision = self._classify(risks, deny_t, ask_t)
        reasons = [f"{qid}={p:.2f}" for qid, p in sorted(risks.items(), key=lambda kv: -kv[1]) if p >= ask_t]
        if error:
            reasons.append(f"primary backend error, used fallback: {error}")
        return Verdict(
            decision, source, reasons, risks=risks, signals=signals, backend=resp.backend, model=resp.model,
            latency_ms=elapsed, model_latency_ms=resp.latency_ms, input_tokens=resp.input_tokens,
            cost_usd=resp.cost_usd,
        )


def _schema_descriptions(schema: Any) -> str:
    """Parameter descriptions are part of the tool's prompt surface too."""
    out: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            desc = node.get("description")
            if isinstance(desc, str) and desc.strip():
                out.append(f"{path}: {desc}")
            for key, child in (node.get("properties") or {}).items():
                walk(child, f"{path}.{key}" if path else key)
            if isinstance(node.get("items"), dict):
                walk(node["items"], f"{path}[]")

    walk(schema, "")
    return "\n".join(out)
