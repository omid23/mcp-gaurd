"""Offline rule-based backend.

It answers the same question ids as the model backends using only regex
signals. It exists as the zero-cost baseline in evals, as the fallback when a
hosted model is unreachable, and so the whole gateway runs without API keys.
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

from ..policy import (
    STRONG_DESCRIPTION,
    STRONG_OUTPUT,
    call_signals,
    description_signals,
    output_signals,
)
from ..protocol import canonical_json
from .models import Answer, DecisionResponse, Question

_DESTRUCTIVE_VERBS = re.compile(
    r"(?:^|_|-)(?:delete|remove|rm|drop|destroy|purge|wipe|truncate|kill|terminate|revoke|erase|reset)(?:_|-|$)",
    re.I,
)
_WRITE_VERBS = re.compile(r"(?:^|_|-)(?:write|overwrite|edit|move|rename|update|replace|put|set)(?:_|-|$)", re.I)
_SEND_VERBS = re.compile(r"(?:^|_|-)(?:send|post|upload|email|mail|publish|share|forward|webhook|notify|message)", re.I)
_SYSTEM_PATH = re.compile(r"(?:^|[\"'\s])(?:/etc/|/boot/|/usr/|/bin/|/sbin/|/var/lib/|~/\.\w+rc\b|C:\\\\Windows)", re.I)
_SECRET_WORDS = re.compile(r"\b(?:password|passwd|secret|token|credential|private[ _]key|api[ _]?key)s?\b", re.I)
_SALIENT = re.compile(r"(?:https?://\S+|[\w.+-]+@[\w.-]+\.\w+|(?:~|\.{0,2})?/[\w.\-/~]+|\b[\w\-]+\.(?:txt|json|ya?ml|csv|md|py|js|ts|log|env|ini|conf|pem|key)\b)")


def _prob_from_count(n: int, table: tuple[float, ...]) -> float:
    return table[min(n, len(table) - 1)]


class HeuristicBackend:
    name = "heuristic"

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> DecisionResponse:
        started = time.perf_counter()
        kind = state.get("kind") if isinstance(state, dict) else None
        if kind == "tool_call":
            probs = self._call(state)
        elif kind == "tool_description":
            probs = self._description(state)
        elif kind == "tool_output":
            probs = self._output(state)
        else:
            probs = {}
        answers = {qid: Answer("noul", noul=probs.get(qid, 0.05)) for qid in questions}
        return DecisionResponse(
            answers=answers,
            backend=self.name,
            model="regex-v1",
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _call(self, state: dict) -> dict[str, float]:
        tool = state.get("tool") or {}
        name = str(tool.get("name", ""))
        annotations = tool.get("annotations") or {}
        args = state.get("arguments") or {}
        args_text = canonical_json(args)
        signals = call_signals(args, [])
        # Egress is judged by the policy layer, which knows the allowlist and the user's
        # intent; reuse its verdict instead of recomputing it without that context.
        egress = (state.get("deterministic_signals") or {}).get("external_egress")
        if egress:
            signals["external_egress"] = egress
        else:
            signals.pop("external_egress", None)
        intent = state.get("user_intent")

        destructive = 0.05
        if signals.get("dangerous_command"):
            destructive = 0.95
        elif _DESTRUCTIVE_VERBS.search(name):
            destructive = 0.7
        elif _WRITE_VERBS.search(name) and (signals.get("sensitive_path") or _SYSTEM_PATH.search(args_text)):
            destructive = 0.88
        elif annotations.get("destructiveHint") is True:
            destructive = 0.35

        credential = 0.03
        if signals.get("sensitive_path") or signals.get("secret_in_args"):
            credential = 0.95
        elif _SECRET_WORDS.search(args_text):
            credential = 0.6

        exfil = 0.04
        if signals.get("external_egress"):
            exfil = 0.45
            if _SEND_VERBS.search(name):
                exfil = 0.65
            if credential >= 0.6:
                exfil = 0.92

        privilege = 0.9 if signals.get("privilege_change") else 0.03

        probs = {
            "destructive": destructive,
            "exfiltration": exfil,
            "credential_access": credential,
            "privilege_escalation": privilege,
            "purpose_mismatch": 0.1,
        }
        if isinstance(intent, str) and intent.strip():
            probs["intent_mismatch"] = self._intent_mismatch(intent, args, signals)
        return probs

    @staticmethod
    def _intent_mismatch(intent: str, args: Any, signals: dict) -> float:
        intent_l = intent.lower()
        salient = set()
        for value in _iter_strings(args):
            for m in _SALIENT.finditer(value):
                token = m.group(0).strip(".,;:'\")(")
                if len(token) > 2:
                    salient.add(token)
        if not salient:
            return 0.2

        def mentioned(token: str) -> bool:
            t = token.lower()
            if t in intent_l or t.rsplit("/", 1)[-1] in intent_l:
                return True
            if t.startswith(("http://", "https://")):
                host = (urlparse(t).hostname or "").removeprefix("www.")
                return bool(host) and host in intent_l
            return False

        missing = [s for s in salient if not mentioned(s)]
        if not missing:
            return 0.08
        risky = signals.get("sensitive_path") or any(
            not mentioned(d) for d in signals.get("external_egress", [])
        )
        if risky:
            return 0.95
        return 0.7 if len(missing) == len(salient) else 0.55

    def _description(self, state: dict) -> dict[str, float]:
        text = str(state.get("description", ""))
        signals = description_signals(text, state.get("other_tools") or [])
        strong = [s for s in signals if s in STRONG_DESCRIPTION]
        weak = [s for s in signals if s not in STRONG_DESCRIPTION and s != "cross_tool_reference"]
        if strong:
            hidden = 0.96
        else:
            hidden = _prob_from_count(len(weak), (0.04, 0.55, 0.78, 0.9))
        cross = 0.04
        if "cross_tool_reference" in signals:
            directive = {"precondition", "mandatory_step", "agent_directive"} & set(signals)
            cross = 0.88 if directive else 0.35
        return {"hidden_instructions": hidden, "cross_tool_manipulation": cross}

    def _output(self, state: dict) -> dict[str, float]:
        signals = output_signals(str(state.get("output", "")))
        strong = [s for s in signals if s in STRONG_OUTPUT or s == "hidden_unicode"]
        weak = [s for s in signals if s not in strong]
        if strong:
            p = 0.93
        else:
            p = _prob_from_count(len(weak), (0.03, 0.5, 0.78, 0.9))
        return {"injection": p}

    async def aclose(self) -> None:
        return None


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _iter_strings(v)
