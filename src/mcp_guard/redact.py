"""Secret detection and redaction.

Secrets are detected locally so the deterministic policy can block them, and
they are masked before any state is sent to a hosted decision model.
"""

from __future__ import annotations

import re
from typing import Any

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"
    ),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "github_pat": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{24,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "stripe_key": re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
    "password_assignment": re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"]?[^\s'\",;]{6,}"
    ),
}


def find_secrets(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for kind, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            found.append((kind, match.group(0)))
    return found


def redact_text(text: str) -> str:
    for kind, pattern in SECRET_PATTERNS.items():
        text = pattern.sub(f"[REDACTED:{kind}]", text)
    return text


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    return value
