"""Small local prompt-injection classifier backend (optional extra: ``classifier``).

Independent guardrail benchmarks have found that a ~200M-parameter
fine-tuned classifier can match much larger judges on prompt injection, so
this is the cheapest serious baseline to compare Jev against. It answers the
description and output questions; tool-call questions fall back to the
heuristic backend because a single injection label says nothing about, say,
privilege escalation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..config import ClassifierConfig
from .heuristic import HeuristicBackend
from .models import Answer, DecisionError, DecisionResponse, Question

TEXT_KINDS = {"tool_description": "description", "tool_output": "output"}


class ClassifierBackend:
    name = "classifier"

    def __init__(self, config: ClassifierConfig, pipeline: Any = None):
        self.config = config
        self._pipeline = pipeline
        self._fallback = HeuristicBackend()

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise DecisionError("install the 'classifier' extra to use this backend") from exc
            self._pipeline = pipeline(
                "text-classification", model=self.config.model, device=self.config.device, truncation=True
            )
        return self._pipeline

    def _p_injection(self, text: str) -> float:
        result = self._load()(text, top_k=None)
        if result and isinstance(result[0], list):
            result = result[0]
        for item in result:
            if str(item.get("label", "")).upper() == self.config.injection_label.upper():
                return float(item["score"])
        return 0.0

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> DecisionResponse:
        kind = state.get("kind") if isinstance(state, dict) else None
        field = TEXT_KINDS.get(kind or "")
        if field is None:
            resp = await self._fallback.evaluate(state, questions)
            resp.backend = self.name
            resp.extra["delegated"] = "heuristic"
            return resp
        started = time.perf_counter()
        p = await asyncio.to_thread(self._p_injection, str(state.get(field, "")))
        answers = {qid: Answer("noul", noul=p) for qid in questions}
        return DecisionResponse(
            answers=answers, backend=self.name, model=self.config.model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def aclose(self) -> None:
        return None
