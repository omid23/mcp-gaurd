"""Jev (TypeSafe AI) backend over the /v1/systemone wire format.

The same client works against any server that speaks this format, including
local open-weight decision models, by changing ``base_url``.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from ..config import JevConfig
from .models import Answer, DecisionError, DecisionResponse, Question

RETRY_STATUS = {429, 500, 502, 503, 504, 529}


class JevBackend:
    name = "jev"

    def __init__(self, config: JevConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.timeout_s)

    @property
    def url(self) -> str:
        return self.config.base_url.rstrip("/") + "/v1/systemone"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> DecisionResponse:
        body = {
            "model": self.config.model,
            "state": state,
            "questions": {qid: q.to_api() for qid, q in questions.items()},
        }
        started = time.perf_counter()
        last_error: str = "unknown error"
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self._client.post(
                    self.url, json=body, headers=self._headers(), timeout=self.config.timeout_s
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json(), questions, started)
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRY_STATUS:
                    break
            if attempt < self.config.max_retries:
                await asyncio.sleep(0.2 * (2**attempt))
        raise DecisionError(f"jev request failed: {last_error}")

    def _parse(self, data: dict, questions: dict[str, Question], started: float) -> DecisionResponse:
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, dict):
            raise DecisionError("jev response has no answers object")
        answers: dict[str, Answer] = {}
        for qid in questions:
            a = raw_answers.get(qid)
            if not isinstance(a, dict):
                raise DecisionError(f"jev response missing answer for '{qid}'")
            kind = a.get("type")
            if kind == "noul":
                answers[qid] = Answer("noul", noul=float(a["noul"]))
            elif kind == "choice":
                answers[qid] = Answer(
                    "choice",
                    choice=a.get("choice"),
                    probabilities={k: float(v) for k, v in (a.get("probabilities") or {}).items()},
                    confidence=a.get("confidence"),
                )
            else:
                raise DecisionError(f"unsupported answer type '{kind}' for '{qid}'")
        usage = data.get("usage") or {}
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        cost = (
            in_tok * self.config.input_cost_per_mtok + out_tok * self.config.output_cost_per_mtok
        ) / 1_000_000
        return DecisionResponse(
            answers=answers,
            backend=self.name,
            model=str(data.get("model") or self.config.model),
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()
