"""Surgical removal of injected passages from tool output.

The ``quarantine`` output action withholds a whole result when any part of it
looks like instructions aimed at the assistant. That is safe but costly: one
injected sentence in a long email or web page hides everything else. The
``sanitize`` action instead splits the text into passages (sentences, lines and
``<tag>…</tag>`` blocks), removes the ones that carry injection signals, and
strips invisible characters. The gateway then re-scans what is left and falls
back to quarantine if it still looks suspicious, so sanitizing can only narrow
what is withheld, never let more through.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from .policy import output_signals, strip_hidden_unicode

REMOVED_MARKER = "[mcp-guard removed a passage that addressed the AI assistant]"

# A tagged block is removed whole, so an untagged sentence inside it cannot survive.
_TAG_BLOCK = re.compile(
    r"<\s*(important|system|instructions?|admin|secret|hidden)\s*>[\s\S]*?<\s*/\s*\1\s*>", re.I
)
_PASSAGE_BREAK = re.compile(r"(?<=[.!?])\s+|\n")


@dataclass
class SanitizeReport:
    passages_removed: int = 0
    hidden_chars_removed: int = 0
    signals: set[str] = field(default_factory=set)

    @property
    def changed(self) -> bool:
        return bool(self.passages_removed or self.hidden_chars_removed)

    def merge(self, other: SanitizeReport) -> None:
        self.passages_removed += other.passages_removed
        self.hidden_chars_removed += other.hidden_chars_removed
        self.signals |= other.signals

    def to_meta(self) -> dict:
        meta: dict[str, Any] = {"passagesRemoved": self.passages_removed, "signals": sorted(self.signals)}
        if self.hidden_chars_removed:
            meta["hiddenCharsRemoved"] = self.hidden_chars_removed
        return meta


def _passages(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans, pos = [], start
    for m in _PASSAGE_BREAK.finditer(text, start, end):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < end:
        spans.append((pos, end))
    return spans


def _units(text: str) -> list[tuple[int, int]]:
    units, pos = [], 0
    for m in _TAG_BLOCK.finditer(text):
        units.extend(_passages(text, pos, m.start()))
        units.append((m.start(), m.end()))
        pos = m.end()
    units.extend(_passages(text, pos, len(text)))
    return units


def sanitize_text(text: str) -> tuple[str, SanitizeReport]:
    report = SanitizeReport()
    stripped = strip_hidden_unicode(text)
    report.hidden_chars_removed = len(text) - len(stripped)
    if report.hidden_chars_removed:
        report.signals.add("hidden_unicode")

    out, pos, last_removed = [], 0, False
    for start, end in _units(stripped):
        signals = output_signals(stripped[start:end])
        if not signals:
            out.append(stripped[pos:end])
            last_removed = False
        else:
            report.passages_removed += 1
            report.signals.update(signals)
            if not last_removed:  # one marker per run of adjacent removed passages
                out.append(stripped[pos:start])
                out.append(REMOVED_MARKER)
            last_removed = True
        pos = end
    out.append(stripped[pos:])
    return "".join(out), report


def _sanitize_strings(value: Any, report: SanitizeReport) -> Any:
    if isinstance(value, str):
        clean, part = sanitize_text(value)
        report.merge(part)
        return clean
    if isinstance(value, list):
        return [_sanitize_strings(v, report) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_strings(v, report) for k, v in value.items()}
    return value


def _sanitize_item(item: Any, report: SanitizeReport) -> None:
    if not isinstance(item, dict):
        return
    if isinstance(item.get("text"), str):
        item["text"], part = sanitize_text(item["text"])
        report.merge(part)
    resource = item.get("resource")
    if isinstance(resource, dict) and isinstance(resource.get("text"), str):
        resource["text"], part = sanitize_text(resource["text"])
        report.merge(part)


def sanitize_result(result: dict) -> tuple[dict, SanitizeReport]:
    """Return a sanitized copy of a tools/call or resources/read result.

    Covers every place :func:`protocol.extract_text` reads from, so the re-scan
    sees exactly what the model would.
    """
    clean = copy.deepcopy(result)
    report = SanitizeReport()
    for key in ("content", "contents"):
        for item in clean.get(key) or []:
            _sanitize_item(item, report)
    if "structuredContent" in clean:
        clean["structuredContent"] = _sanitize_strings(clean["structuredContent"], report)
    return clean, report
