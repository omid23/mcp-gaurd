"""Eval case schema and JSONL loading."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

Kind = Literal["tool_call", "tool_description", "tool_output"]


@dataclass
class EvalCase:
    id: str
    kind: Kind
    label: Literal["attack", "benign"]
    category: str
    server: str
    tool: dict[str, Any]
    arguments: dict[str, Any] | None = None
    output: str | None = None
    intent: str | None = None
    siblings: list[str] = field(default_factory=list)
    split: str = "test"
    source: str = "handcrafted"

    @classmethod
    def from_dict(cls, d: dict) -> EvalCase:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {})}


def load_jsonl(path: str | Path) -> list[EvalCase]:
    cases = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def write_jsonl(path: str | Path, cases: Iterable[EvalCase]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def builtin_path(name: str) -> Path:
    return Path(str(resources.files("mcp_guard.eval") / "data" / f"{name}.jsonl"))


def load_dataset(spec: str) -> list[EvalCase]:
    """``sample`` (built-in), or a path to a JSONL file such as data/mcptox/calls.jsonl."""
    path = Path(spec)
    if path.exists():
        return load_jsonl(path)
    builtin = builtin_path(spec)
    if builtin.exists():
        return load_jsonl(builtin)
    raise FileNotFoundError(f"dataset '{spec}' not found (built-in names: sample)")
