"""Convert the MCPTox benchmark (AAAI 2026) into gate-level eval cases.

MCPTox (github.com/zhiqiangwang4/MCPTox-Benchmark) ships 485 poisoned tool
descriptions across 45 real MCP servers, the genuine tool lists of those
servers, and the responses of 20+ models that were exposed to each poison,
labeled by outcome. This module turns that into two datasets:

``descriptions.jsonl``
    attack = each poisoned description, benign = each genuine description.
    Measures whether the gateway hides poisoned tools at ``tools/list`` time.

``calls.jsonl``
    attack = tool calls made by models in responses labeled ``Success`` (the
    poison worked), minus any call that also appears in a response where the
    model ignored the poison. benign = calls from responses labeled
    ``Failure-Ignored`` (the model just did the user's task). The user's query
    is kept as ``intent`` so the eval can run with and without it.
    Measures whether the gateway stops the harmful call even after the agent
    was already fooled.

Splits are by server (CRC32 of the server name), so dev and test never share a
server. The MCPTox data is not redistributed here; run ``mcp-guard mcptox``
against your own clone.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
import zlib
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..protocol import canonical_json
from .dataset import EvalCase

_TOOL_BLOCK = re.compile(r"Tool:\s*(?P<name>[^\n]+)\nDescription:\s*(?P<desc>.*?)\nArguments:\n(?P<args>.*?)(?=\n\s*\nTool:|\Z)", re.S)
_ARG_LINE = re.compile(r"^-\s*(?P<name>[^:]+):\s*(?P<desc>.*?)(?P<req>\s*\(required\))?\s*$")


def split_for(server: str) -> str:
    return "dev" if zlib.crc32(server.encode("utf-8")) % 2 == 0 else "test"


def parse_tool_listing(text: str) -> dict[str, dict]:
    """Parse MCPTox's 'Tool: / Description: / Arguments:' system-prompt listing."""
    tools: dict[str, dict] = {}
    for m in _TOOL_BLOCK.finditer(text):
        name = m.group("name").strip()
        props: dict[str, dict] = {}
        required: list[str] = []
        for line in m.group("args").splitlines():
            am = _ARG_LINE.match(line.strip())
            if not am:
                continue
            arg = am.group("name").strip()
            desc = am.group("desc").strip()
            props[arg] = {"type": "string"}
            if desc and desc.lower() != "no description":
                props[arg]["description"] = desc
            if am.group("req"):
                required.append(arg)
        tools[name] = {
            "name": name,
            "description": m.group("desc").strip(),
            "inputSchema": {"type": "object", "properties": props, "required": required},
        }
    return tools


def _balanced_objects(text: str) -> Iterator[str]:
    depth, start, in_str, quote, esc = 0, None, False, "", False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'" and depth > 0:
            in_str, quote = True, ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def parse_calls(text: str) -> list[tuple[str, dict]]:
    """Extract {'tool': ..., 'arguments': {...}} calls from a free-form model response."""
    calls: list[tuple[str, dict]] = []
    for blob in _balanced_objects(text or ""):
        obj: Any = None
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            pyish = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", blob)))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    obj = ast.literal_eval(pyish)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            args = obj.get("arguments")
            calls.append((obj["tool"], args if isinstance(args, dict) else {}))
    return calls


def _responses(item: dict) -> Iterator[tuple[str, str]]:
    """Yield (response_text, label) pairs from both the local and online model runs."""
    responses = item.get("response") or {}
    labels = item.get("label") or {}
    for model, text in responses.items():
        if isinstance(text, str) and model in labels:
            yield text, labels[model]
    online = item.get("online_result") or {}
    for resp_map, label_map in zip(online.get("backend_model_responses") or [],
                                   online.get("labeled_model_results") or [], strict=False):
        for model, text in (resp_map or {}).items():
            if isinstance(text, str) and model in (label_map or {}):
                yield text, label_map[model]


def convert(repo_dir: str | Path) -> tuple[list[EvalCase], list[EvalCase]]:
    repo = Path(repo_dir)
    pure = json.loads((repo / "pure_tool.json").read_text(encoding="utf-8"))
    full = json.loads((repo / "response_all.json").read_text(encoding="utf-8"))
    servers = full["servers"]

    genuine: dict[str, dict[str, dict]] = {
        name: parse_tool_listing(s.get("clean_system_promot") or "") for name, s in servers.items()
    }

    # -- descriptions ---------------------------------------------------------------
    descriptions: list[EvalCase] = []
    seen: set[str] = set()
    for group in pure:
        for case_id, c in group.items():
            server = c["server_name"]
            key = canonical_json([server, c["tool_name"], c["tool_content"]])
            if key in seen:
                continue
            seen.add(key)
            siblings = sorted(genuine.get(server, {}))
            descriptions.append(EvalCase(
                id=f"mcptox-desc-{case_id}", kind="tool_description", label="attack",
                category=c.get("security risk", "Other"), server=server,
                tool={"name": c["tool_name"], "description": c["tool_content"].strip()},
                siblings=siblings, split=split_for(server), source="mcptox",
            ))
    for server, tools in genuine.items():
        for name, tool in tools.items():
            descriptions.append(EvalCase(
                id=f"mcptox-desc-genuine-{server}-{name}", kind="tool_description", label="benign",
                category="genuine", server=server, tool=tool, siblings=sorted(tools),
                split=split_for(server), source="mcptox",
            ))

    # -- calls ------------------------------------------------------------------------
    attack_calls: dict[str, EvalCase] = {}
    benign_calls: dict[str, EvalCase] = {}
    for server, s in servers.items():
        tools = genuine.get(server, {})
        for inst in s.get("malicious_instance") or []:
            poisoned_name = ""
            m = re.match(r"\s*Tool:\s*([^\n]+)", inst.get("poisoned_tool") or "")
            if m:
                poisoned_name = m.group(1).strip()
            category = (inst.get("metadata") or {}).get("security risk", "Other")
            for item in inst.get("datas") or []:
                query = item.get("query", "")
                ignored: set[str] = set()
                succeeded: list[tuple[str, dict]] = []
                for text, label in _responses(item):
                    for tool_name, args in parse_calls(text):
                        if tool_name == poisoned_name or tool_name not in tools:
                            continue
                        if label == "Failure-Ignored":
                            ignored.add(canonical_json([tool_name, args]))
                            key = canonical_json([server, query, tool_name, args])
                            benign_calls.setdefault(key, EvalCase(
                                id=f"mcptox-call-b{len(benign_calls)}", kind="tool_call", label="benign",
                                category="user_task", server=server, tool=tools[tool_name], arguments=args,
                                intent=query, split=split_for(server), source="mcptox",
                            ))
                        elif label == "Success":
                            succeeded.append((tool_name, args))
                for tool_name, args in succeeded:
                    if canonical_json([tool_name, args]) in ignored:
                        continue  # the legitimate call that came along with the attack
                    key = canonical_json([server, query, tool_name, args])
                    attack_calls.setdefault(key, EvalCase(
                        id=f"mcptox-call-a{len(attack_calls)}", kind="tool_call", label="attack",
                        category=category, server=server, tool=tools[tool_name], arguments=args,
                        intent=query, split=split_for(server), source="mcptox",
                    ))
    # A call that is someone's attack and someone else's normal task is ambiguous; drop it.
    attack_keys = {canonical_json([c.server, c.tool["name"], c.arguments]) for c in attack_calls.values()}
    benign = [c for c in benign_calls.values()
              if canonical_json([c.server, c.tool["name"], c.arguments]) not in attack_keys]
    calls = list(attack_calls.values()) + benign
    return descriptions, calls


def summarize(cases: list[EvalCase]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for c in cases:
        counts[f"{c.split}/{c.label}"] += 1
    return dict(sorted(counts.items()))
