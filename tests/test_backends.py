from __future__ import annotations

import json

import httpx
import pytest

from mcp_guard.config import JevConfig, LLMJudgeConfig, ServerConfig, Settings, VertexConfig
from mcp_guard.guard import (
    Answer,
    DecisionError,
    DecisionResponse,
    GuardEngine,
    HeuristicBackend,
    JevBackend,
    LLMJudgeBackend,
    VertexBackend,
    build_backend,
)
from mcp_guard.guard.questions import CALL_QUESTIONS, OUTPUT_QUESTIONS
from mcp_guard.guard.vertex import ADCTokenProvider


def jev_answer(questions: dict, p: float = 0.1) -> dict:
    return {"model": "jev-1.13.0", "answers": {q: {"type": "noul", "noul": p} for q in questions},
            "usage": {"input_tokens": 1000, "output_tokens": 0}}


async def test_jev_request_shape_and_cost(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=jev_answer(seen["body"]["questions"], 0.2))

    backend = JevBackend(JevConfig(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    resp = await backend.evaluate({"kind": "tool_call"}, CALL_QUESTIONS)
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer ts-key"
    q = seen["body"]["questions"]["destructive"]
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}
    assert seen["body"]["model"] == "jev-1.13"
    assert resp.answers["destructive"].p_yes == 0.2
    assert resp.cost_usd == pytest.approx(1000 * 0.042 / 1e6)


async def test_jev_retries_overload_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(529, text="overloaded")
        return httpx.Response(200, json=jev_answer(json.loads(request.content)["questions"]))

    backend = JevBackend(JevConfig(max_retries=2), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await backend.evaluate({}, OUTPUT_QUESTIONS)
    assert calls["n"] == 2


async def test_jev_auth_error_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    backend = JevBackend(JevConfig(max_retries=3), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(DecisionError, match="401"):
        await backend.evaluate({}, OUTPUT_QUESTIONS)
    assert calls["n"] == 1


async def test_jev_missing_answer_is_error():
    backend = JevBackend(JevConfig(), httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"answers": {}}))))
    with pytest.raises(DecisionError, match="missing answer"):
        await backend.evaluate({}, OUTPUT_QUESTIONS)


async def test_llm_judge_parses_probabilities():
    def handler(request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        content = "```json\n" + json.dumps({"injection": 0.91}) + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 500, "completion_tokens": 10}})

    backend = LLMJudgeBackend(LLMJudgeConfig(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    resp = await backend.evaluate({"kind": "tool_output", "output": "x"}, OUTPUT_QUESTIONS)
    assert resp.answers["injection"].p_yes == 0.91
    assert resp.input_tokens == 500


class FixedBackend:
    name = "fixed"

    def __init__(self, p: float = 0.0, fail: bool = False):
        self.p, self.fail, self.calls = p, fail, 0

    async def evaluate(self, state, questions):
        self.calls += 1
        if self.fail:
            raise DecisionError("down")
        return DecisionResponse({q: Answer("noul", noul=self.p) for q in questions}, "fixed", "m", 1.0)

    async def aclose(self):
        pass


SERVER = ServerConfig(url="http://x")
TOOL = {"name": "lookup", "description": "Look something up."}


@pytest.mark.parametrize("p,expected", [(0.1, "allow"), (0.6, "ask"), (0.9, "deny")])
async def test_thresholds(p, expected):
    engine = GuardEngine(Settings(), FixedBackend(p))
    assert (await engine.check_call("s", SERVER, TOOL, {"q": "x"})).decision == expected


async def test_fail_mode_without_fallback():
    s = Settings()
    s.gateway.fail_mode = "deny"
    v = await GuardEngine(s, FixedBackend(fail=True)).check_call("s", SERVER, TOOL, {"q": "x"})
    assert v.decision == "deny" and v.source == "fail_mode"


async def test_fallback_used_when_primary_fails():
    v = await GuardEngine(Settings(), FixedBackend(fail=True), fallback=HeuristicBackend()).check_call(
        "s", SERVER, TOOL, {"q": "x"})
    assert v.source == "fallback" and v.decision == "allow"


async def test_scans_fail_open():
    v = await GuardEngine(Settings(), FixedBackend(fail=True)).check_output("s", "t", "plain text")
    assert v.decision == "allow" and v.source == "fail_mode"


async def test_cache_hits_skip_model():
    backend = FixedBackend(0.1)
    engine = GuardEngine(Settings(), backend)
    await engine.check_call("s", SERVER, TOOL, {"q": "x"})
    await engine.check_call("s", SERVER, TOOL, {"q": "x"})
    assert backend.calls == 1


async def test_secrets_redacted_before_model():
    seen = {}

    class Capture(FixedBackend):
        async def evaluate(self, state, questions):
            seen["state"] = state
            return await super().evaluate(state, questions)

    s = Settings()
    s.policy.block_secrets_in_args = False
    await GuardEngine(s, Capture()).check_call("s", SERVER, TOOL, {"q": "token AKIAABCDEFGHIJKLMNOP"})
    assert "AKIA" not in json.dumps(seen["state"])


async def test_intent_question_added_only_with_intent():
    seen = []

    class Capture(FixedBackend):
        async def evaluate(self, state, questions):
            seen.append(set(questions))
            return await super().evaluate(state, questions)

    engine = GuardEngine(Settings(), Capture())
    await engine.check_call("s", SERVER, TOOL, {"q": "a"})
    await engine.check_call("s", SERVER, TOOL, {"q": "b"}, intent="look up b")
    assert "intent_mismatch" not in seen[0] and "intent_mismatch" in seen[1]


# -- vertex (Gemini on Google Cloud) --------------------------------------------------


def gemini_reply(probs: dict, **usage) -> dict:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(probs)}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 800, "candidatesTokenCount": 20, **usage},
        "modelVersion": "gemini-3.5-flash-lite-001",
    }


async def fake_token() -> str:
    return "adc-token"


def vertex(handler, config: VertexConfig | None = None, **kw) -> VertexBackend:
    return VertexBackend(config or VertexConfig(), httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                         token_provider=kw.get("token_provider", fake_token))


@pytest.fixture
def no_google_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)


async def test_vertex_express_mode_request_and_cost(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    seen = {}

    def handler(request):
        seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
        return httpx.Response(200, json=gemini_reply({"injection": 0.88}, thoughtsTokenCount=30))

    cfg = VertexConfig(input_cost_per_mtok=0.1, output_cost_per_mtok=0.4)
    resp = await vertex(handler, cfg).evaluate({"kind": "tool_output", "output": "x"}, OUTPUT_QUESTIONS)
    assert seen["url"] == ("https://aiplatform.googleapis.com/v1/publishers/google/models/"
                           "gemini-3.5-flash-lite:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "g-key" and "authorization" not in seen["headers"]
    gen = seen["body"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json" and gen["temperature"] == 0
    assert gen["responseSchema"]["required"] == ["injection"]
    assert {s["threshold"] for s in seen["body"]["safetySettings"]} == {"BLOCK_NONE"}
    assert resp.answers["injection"].p_yes == 0.88
    assert resp.backend == "vertex" and resp.model == "gemini-3.5-flash-lite-001"
    assert resp.input_tokens == 800 and resp.output_tokens == 50  # thinking tokens count as output
    assert resp.cost_usd == pytest.approx((800 * 0.1 + 50 * 0.4) / 1e6)


async def test_vertex_adc_regional_endpoint(no_google_env):
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        body = json.loads(request.content)
        assert "safetySettings" not in body
        return httpx.Response(200, json=gemini_reply({q: 0.1 for q in CALL_QUESTIONS}))

    cfg = VertexConfig(project="acme-sec", location="europe-west4", safety_threshold=None)
    resp = await vertex(handler, cfg).evaluate({"kind": "tool_call"}, CALL_QUESTIONS)
    assert seen["url"] == ("https://europe-west4-aiplatform.googleapis.com/v1/projects/acme-sec/locations/"
                           "europe-west4/publishers/google/models/gemini-3.5-flash-lite:generateContent")
    assert seen["auth"] == "Bearer adc-token"
    assert set(resp.answers) == set(CALL_QUESTIONS)


async def test_vertex_project_from_credentials(no_google_env):
    class Provider:
        project = None

        async def __call__(self):
            self.project = "from-adc"
            return "t"

    def handler(request):
        assert "/projects/from-adc/locations/global/" in str(request.url)
        return httpx.Response(200, json=gemini_reply({"injection": 0.2}))

    resp = await vertex(handler, token_provider=Provider()).evaluate({}, OUTPUT_QUESTIONS)
    assert resp.answers["injection"].p_yes == 0.2


async def test_vertex_without_project_is_error(no_google_env):
    backend = vertex(lambda r: httpx.Response(200, json=gemini_reply({"injection": 0.2})))
    with pytest.raises(DecisionError, match="project"):
        await backend.evaluate({}, OUTPUT_QUESTIONS)


async def test_vertex_blocked_prompt_is_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    backend = vertex(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(DecisionError, match="SAFETY"):
        await backend.evaluate({}, OUTPUT_QUESTIONS)


async def test_vertex_ignores_thought_parts(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    reply = gemini_reply({"injection": 0.7})
    reply["candidates"][0]["content"]["parts"].insert(0, {"text": "thinking {not json", "thought": True})
    resp = await vertex(lambda r: httpx.Response(200, json=reply)).evaluate({}, OUTPUT_QUESTIONS)
    assert resp.answers["injection"].p_yes == 0.7


async def test_vertex_retries_rate_limit_but_not_permission_errors(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    calls = []

    def flaky(request):
        calls.append(1)
        return httpx.Response(429) if len(calls) == 1 else httpx.Response(200, json=gemini_reply({"injection": 0.3}))

    resp = await vertex(flaky, VertexConfig(max_retries=2)).evaluate({}, OUTPUT_QUESTIONS)
    assert resp.answers["injection"].p_yes == 0.3 and len(calls) == 2

    denied = []
    backend = vertex(lambda r: denied.append(1) or httpx.Response(403, text="PERMISSION_DENIED"),
                     VertexConfig(max_retries=3))
    with pytest.raises(DecisionError, match="403"):
        await backend.evaluate({}, OUTPUT_QUESTIONS)
    assert len(denied) == 1


async def test_adc_token_provider_refreshes_expired_credentials(monkeypatch):
    google_auth = pytest.importorskip("google.auth")

    class Creds:
        valid, token, refreshes = False, None, 0

        def refresh(self, request):
            self.refreshes += 1
            self.valid, self.token = True, f"tok-{self.refreshes}"

    creds = Creds()
    monkeypatch.setattr(google_auth, "default", lambda scopes: (creds, "adc-project"))
    provider = ADCTokenProvider()
    assert await provider() == "tok-1" and provider.project == "adc-project"
    assert await provider() == "tok-1"  # still valid: no second refresh
    creds.valid = False
    assert await provider() == "tok-2"


async def test_adc_token_provider_wraps_auth_errors(monkeypatch):
    google_auth = pytest.importorskip("google.auth")
    from google.auth.exceptions import DefaultCredentialsError

    def missing(scopes):
        raise DefaultCredentialsError("no credentials found")

    monkeypatch.setattr(google_auth, "default", missing)
    with pytest.raises(DecisionError, match="credentials unavailable"):
        await ADCTokenProvider()()


def test_build_vertex_backend():
    assert isinstance(build_backend("vertex", Settings()), VertexBackend)
