from __future__ import annotations

import json

import httpx
import pytest

from mcp_guard.config import JevConfig, LLMJudgeConfig, ServerConfig, Settings
from mcp_guard.guard import (
    Answer,
    DecisionError,
    DecisionResponse,
    GuardEngine,
    HeuristicBackend,
    JevBackend,
    LLMJudgeBackend,
)
from mcp_guard.guard.questions import CALL_QUESTIONS, OUTPUT_QUESTIONS


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
