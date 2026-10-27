from __future__ import annotations

import httpx

from ..config import Settings
from .classifier import ClassifierBackend
from .engine import GuardEngine, Verdict
from .heuristic import HeuristicBackend
from .jev import JevBackend
from .llm_judge import LLMJudgeBackend
from .models import Answer, DecisionBackend, DecisionError, DecisionResponse, Question
from .vertex import VertexBackend

__all__ = [
    "Answer", "ClassifierBackend", "DecisionBackend", "DecisionError", "DecisionResponse", "GuardEngine",
    "HeuristicBackend", "JevBackend", "LLMJudgeBackend", "Question", "Verdict", "VertexBackend", "build_backend",
]


def build_backend(name: str, settings: Settings, client: httpx.AsyncClient | None = None) -> DecisionBackend:
    dc = settings.decision
    if name == "jev":
        return JevBackend(dc.jev, client)
    if name == "llm":
        return LLMJudgeBackend(dc.llm, client)
    if name == "vertex":
        return VertexBackend(dc.vertex, client)
    if name == "classifier":
        return ClassifierBackend(dc.classifier)
    if name == "heuristic":
        return HeuristicBackend()
    raise ValueError(f"unknown decision backend '{name}'")
