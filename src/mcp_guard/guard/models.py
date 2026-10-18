"""Backend-neutral types for typed decision questions.

The question shape mirrors TypeSafe's System One API (state + typed questions
in, probabilities out) so the same questions run on Jev, on any server that
speaks /v1/systemone, on an LLM judge, or on the offline heuristic baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class DecisionError(Exception):
    """The decision backend could not produce an answer."""


@dataclass(frozen=True)
class Question:
    type: Literal["noul", "choice"]
    instructions: str
    criteria: Any = None

    def to_api(self) -> dict:
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


@dataclass
class Answer:
    type: str
    noul: float | None = None
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None

    @property
    def p_yes(self) -> float:
        if self.noul is not None:
            return float(self.noul)
        raise DecisionError(f"answer of type {self.type} has no yes-probability")


@dataclass
class DecisionResponse:
    answers: dict[str, Answer]
    backend: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class DecisionBackend(Protocol):
    name: str

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> DecisionResponse: ...

    async def aclose(self) -> None: ...
