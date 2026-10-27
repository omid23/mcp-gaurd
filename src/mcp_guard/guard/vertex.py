"""Gemini on Google Cloud Vertex AI as a judge (optional extra: ``vertex``).

It asks the same typed questions as the other backends through Vertex AI's
``generateContent`` REST API, with a response schema that forces one number
per question id. Two ways to authenticate:

- an API key (Vertex AI express mode) from ``api_key_env``, sent in the
  ``x-goog-api-key`` header so it never appears in a URL or access log;
- otherwise Application Default Credentials via ``google-auth``: Workload
  Identity on GKE, an attached service account on Cloud Run / GCE, or
  ``gcloud auth application-default login`` on a laptop.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from ..config import VertexConfig
from .llm_judge import SYSTEM_PROMPT, build_prompt, parse_probabilities
from .models import DecisionError, DecisionResponse, Question

RETRY_STATUS = {429, 500, 502, 503, 504}
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
HARM_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
)

TokenProvider = Callable[[], Awaitable[str]]


class ADCTokenProvider:
    """OAuth access tokens from Application Default Credentials, refreshed when expired."""

    def __init__(self) -> None:
        self.project: str | None = None
        self._creds: Any = None
        self._lock = asyncio.Lock()

    async def __call__(self) -> str:
        try:
            import google.auth  # type: ignore
            from google.auth.exceptions import GoogleAuthError  # type: ignore
            from google.auth.transport.requests import Request  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise DecisionError("install the 'vertex' extra for Google Cloud credentials, "
                                "or set an API key for Vertex AI express mode") from exc
        async with self._lock:
            try:
                if self._creds is None:
                    self._creds, self.project = await asyncio.to_thread(google.auth.default, scopes=[SCOPE])
                if not self._creds.valid:
                    await asyncio.to_thread(self._creds.refresh, Request())
            except GoogleAuthError as exc:
                raise DecisionError(f"google cloud credentials unavailable: {exc}") from exc
            return self._creds.token


class VertexBackend:
    name = "vertex"

    def __init__(self, config: VertexConfig, client: httpx.AsyncClient | None = None,
                 token_provider: TokenProvider | None = None):
        self.config = config
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.timeout_s)
        self._token = token_provider or ADCTokenProvider()

    def _host(self) -> str:
        loc = self.config.location
        return "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"

    async def _target(self) -> tuple[str, dict[str, str]]:
        headers = {"Content-Type": "application/json"}
        model = f"publishers/google/models/{self.config.model}:generateContent"
        key = os.environ.get(self.config.api_key_env, "")
        if key:
            headers["x-goog-api-key"] = key
            return f"https://{self._host()}/v1/{model}", headers
        headers["Authorization"] = f"Bearer {await self._token()}"
        project = (self.config.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
                   or getattr(self._token, "project", None))
        if not project:
            raise DecisionError("vertex needs a project: set decision.vertex.project or GOOGLE_CLOUD_PROJECT")
        return f"https://{self._host()}/v1/projects/{project}/locations/{self.config.location}/{model}", headers

    def _body(self, state: Any, questions: dict[str, Question]) -> dict:
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": build_prompt(state, questions)}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {qid: {"type": "NUMBER"} for qid in questions},
                    "required": list(questions),
                },
            },
        }
        if self.config.safety_threshold:
            body["safetySettings"] = [{"category": c, "threshold": self.config.safety_threshold}
                                      for c in HARM_CATEGORIES]
        return body

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> DecisionResponse:
        body = self._body(state, questions)
        started = time.perf_counter()
        last_error = "unknown error"
        for attempt in range(self.config.max_retries + 1):
            url, headers = await self._target()
            try:
                resp = await self._client.post(url, json=body, headers=headers, timeout=self.config.timeout_s)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json(), questions, started)
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRY_STATUS:
                    break
            if attempt < self.config.max_retries:
                await asyncio.sleep(0.25 * (2**attempt))
        raise DecisionError(f"vertex request failed: {last_error}")

    def _parse(self, data: dict, questions: dict[str, Question], started: float) -> DecisionResponse:
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            raise DecisionError(f"vertex blocked the prompt: {blocked}")
        candidates = data.get("candidates") or []
        candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p["text"] for p in parts
                       if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought"))
        if not text:
            raise DecisionError(f"vertex returned no text (finishReason={candidate.get('finishReason')})")
        answers = parse_probabilities(text, questions, "vertex")
        usage = data.get("usageMetadata") or {}
        in_tok = int(usage.get("promptTokenCount") or 0)
        # Thinking tokens are billed as output.
        out_tok = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
        cost = (in_tok * self.config.input_cost_per_mtok + out_tok * self.config.output_cost_per_mtok) / 1e6
        return DecisionResponse(
            answers=answers,
            backend=self.name,
            model=str(data.get("modelVersion") or self.config.model),
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()
