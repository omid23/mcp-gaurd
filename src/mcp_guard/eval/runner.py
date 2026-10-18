"""Run eval cases through the guard and report security and utility together.

Metrics, per backend and per gate (call / description / output):

* attack pass rate (gate-level ASR): attacks the gate allowed through;
* attacks stopped: denied or escalated to a human;
* benign allowed (utility), benign escalated (friction), benign denied (false blocks);
* p50 / p95 decision latency, and cost per 1,000 decisions.

"Ask" is reported separately on both sides: escalating every call would score
0% ASR while being useless, so friction is part of the result.
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from ..config import PolicyConfig, ServerConfig, Settings
from ..guard import GuardEngine, build_backend
from ..policy import PolicyEngine
from .dataset import EvalCase


@dataclass
class CaseResult:
    id: str
    kind: str
    label: str
    category: str
    decision: str
    source: str
    max_risk: float
    latency_ms: float
    model_latency_ms: float
    cost_usd: float
    input_tokens: int
    reasons: list[str]


def _model_only_policy() -> PolicyConfig:
    return PolicyConfig(
        block_sensitive_paths=False, block_secrets_in_args=False, block_dangerous_commands=False,
        ask_on_external_egress=False, ask_on_privilege_change=False,
    )


async def _run_case(engine: GuardEngine, case: EvalCase, use_intent: bool) -> CaseResult:
    if case.kind == "tool_call":
        server = ServerConfig(url="http://eval.invalid")
        v = await engine.check_call(case.server, server, case.tool, case.arguments or {},
                                    case.intent if use_intent else None)
    elif case.kind == "tool_description":
        v = await engine.check_description(case.server, case.tool, case.siblings)
    elif case.kind == "tool_output":
        v = await engine.check_output(case.server, case.tool.get("name", ""), case.output or "")
    else:
        raise ValueError(f"unknown case kind {case.kind}")
    decision = v.decision
    if case.kind != "tool_call" and decision == "ask":
        decision = "allow"  # scans have a single threshold; 'ask' cannot occur, but be safe
    return CaseResult(case.id, case.kind, case.label, case.category, decision, v.source,
                      round(v.max_risk, 4), v.latency_ms, v.model_latency_ms, v.cost_usd,
                      v.input_tokens, v.reasons[:3])


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def summarize(results: list[CaseResult]) -> dict:
    by_kind: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_kind[r.kind].append(r)
    out: dict[str, Any] = {}
    for kind, rs in sorted(by_kind.items()):
        attacks = [r for r in rs if r.label == "attack"]
        benign = [r for r in rs if r.label == "benign"]

        def rate(group: list[CaseResult], decision: str) -> float:
            return round(sum(r.decision == decision for r in group) / len(group), 4) if group else 0.0

        model_lat = [r.model_latency_ms for r in rs if r.source in ("model", "fallback")]
        categories: dict[str, dict] = {}
        for cat in sorted({r.category for r in attacks}):
            group = [r for r in attacks if r.category == cat]
            categories[cat] = {"n": len(group), "passed": rate(group, "allow"),
                               "stopped": round(1 - rate(group, "allow"), 4)}
        sources: dict[str, int] = defaultdict(int)
        for r in rs:
            sources[r.source] += 1
        out[kind] = {
            "n_attack": len(attacks),
            "n_benign": len(benign),
            "attack_pass_rate": rate(attacks, "allow"),
            "attack_denied": rate(attacks, "deny"),
            "attack_asked": rate(attacks, "ask"),
            "benign_allowed": rate(benign, "allow"),
            "benign_asked": rate(benign, "ask"),
            "benign_denied": rate(benign, "deny"),
            "latency_p50_ms": round(_pct([r.latency_ms for r in rs], 50), 2),
            "latency_p95_ms": round(_pct([r.latency_ms for r in rs], 95), 2),
            "model_latency_p50_ms": round(_pct(model_lat, 50), 2),
            "model_latency_p95_ms": round(_pct(model_lat, 95), 2),
            "cost_per_1k_usd": round(1000 * sum(r.cost_usd for r in rs) / len(rs), 5) if rs else 0.0,
            "decided_by": dict(sources),
            "attack_categories": categories,
        }
    return out


def stratified_sample(cases: list[EvalCase], n: int, seed: int = 7) -> list[EvalCase]:
    """Up to n cases, keeping the kind/label mix."""
    if n <= 0 or n >= len(cases):
        return cases
    buckets: dict[tuple[str, str], list[EvalCase]] = defaultdict(list)
    for c in cases:
        buckets[(c.kind, c.label)].append(c)
    rng = random.Random(seed)
    picked: list[EvalCase] = []
    for _key, group in sorted(buckets.items()):
        k = max(1, round(n * len(group) / len(cases)))
        picked.extend(rng.sample(group, min(k, len(group))))
    return picked


async def run_eval(
    cases: list[EvalCase],
    backends: list[str],
    settings: Settings | None = None,
    *,
    use_intent: bool = False,
    model_only: bool = False,
    concurrency: int = 8,
) -> dict:
    settings = (settings or Settings()).model_copy(deep=True)
    settings.decision.cache_size = 0  # every case is a fresh decision
    if model_only:
        settings.policy = _model_only_policy()
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cases": len(cases),
        "use_intent": use_intent,
        "model_only": model_only,
        "thresholds": settings.decision.thresholds.model_dump(),
        "backends": {},
    }
    for name in backends:
        backend = build_backend(name, settings)
        engine = GuardEngine(settings, backend, fallback=None, policy=PolicyEngine(settings.policy),
                             static_checks=not model_only)
        sem = asyncio.Semaphore(concurrency)

        async def one(case: EvalCase, engine: GuardEngine = engine, sem: asyncio.Semaphore = sem) -> CaseResult:
            async with sem:
                return await _run_case(engine, case, use_intent)

        started = time.perf_counter()
        results = await asyncio.gather(*(one(c) for c in cases))
        await backend.aclose()
        report["backends"][name] = {
            "wall_s": round(time.perf_counter() - started, 2),
            "summary": summarize(results),
            "errors": sum(r.source == "fail_mode" for r in results),
            "results": [asdict(r) for r in results],
        }
    return report


def to_markdown(report: dict, title: str = "Eval results") -> str:
    lines = [f"# {title}", "",
             f"{report['n_cases']} cases · intent {'on' if report['use_intent'] else 'off'} · "
             f"{'model only' if report['model_only'] else 'policy + model'} · thresholds "
             f"deny ≥ {report['thresholds']['deny']}, ask ≥ {report['thresholds']['ask']} · {report['generated_at']}", ""]
    kinds = sorted({k for b in report["backends"].values() for k in b["summary"]})
    for kind in kinds:
        lines += [f"## {kind}", "",
                  "| backend | attacks | passed (ASR) | denied | asked | benign | allowed | asked | denied | p50 ms | p95 ms | $/1k |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, b in report["backends"].items():
            s = b["summary"].get(kind)
            if not s:
                continue
            lines.append(
                f"| {name} | {s['n_attack']} | {s['attack_pass_rate']:.1%} | {s['attack_denied']:.1%} | "
                f"{s['attack_asked']:.1%} | {s['n_benign']} | {s['benign_allowed']:.1%} | {s['benign_asked']:.1%} | "
                f"{s['benign_denied']:.1%} | {s['latency_p50_ms']:.1f} | {s['latency_p95_ms']:.1f} | "
                f"{s['cost_per_1k_usd']:.4f} |"
            )
        lines.append("")
        for name, b in report["backends"].items():
            s = b["summary"].get(kind)
            if not s or not s["attack_categories"]:
                continue
            lines += [f"Attacks stopped by category ({name}):", ""]
            lines += [f"- {cat}: {c['stopped']:.0%} of {c['n']}" for cat, c in
                      sorted(s["attack_categories"].items(), key=lambda kv: -kv[1]["n"])]
            lines.append("")
    return "\n".join(lines)
