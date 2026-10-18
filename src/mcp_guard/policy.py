"""Deterministic policy: runs before, and can override, any model judgment.

Hard signals (credential paths, secrets, destructive commands) deny outright.
Soft signals (external egress, privilege changes) escalate to a human. Only
calls with no deterministic outcome reach the decision model.
"""

from __future__ import annotations

import fnmatch
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from .config import PolicyConfig, ServerConfig
from .protocol import canonical_json
from .redact import find_secrets

SENSITIVE_PATH_PATTERNS = [
    r"(?:^|[\s'\"/=~])\.ssh(?:/|\b)",
    r"\bid_(?:rsa|dsa|ecdsa|ed25519)\b",
    r"\.aws/(?:credentials|config)\b",
    r"(?:^|[\s'\"/=])\.env(?:\.[A-Za-z0-9_]+)?(?:$|[\s'\",])",
    r"/etc/(?:shadow|gshadow|sudoers)\b",
    r"\.kube/config\b",
    r"\.(?:npmrc|pypirc|netrc|git-credentials|pgpass)\b",
    r"\.docker/config\.json\b",
    r"\.gnupg/",
    r"\b(?:credentials|secrets?)\.(?:json|ya?ml|toml|ini)\b",
    r"\bwallet\.dat\b",
    r"\.config/gcloud/",
    r"\bkeychain\b",
]
_SENSITIVE_PATH = re.compile("|".join(f"(?:{p})" for p in SENSITIVE_PATH_PATTERNS), re.I)

DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+(?:-[a-zA-Z]*\s+)*-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+(?:-[a-zA-Z]*\s+)*-[a-zA-Z]*f[a-zA-Z]*r",
    r"\brm\s+-rf?\b",
    r"\bmkfs(?:\.\w+)?\b",
    r"\bdd\s+if=\S+\s+of=/dev/",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
    r"\bchmod\s+(?:-R\s+)?0?777\b",
    r"\b(?:curl|wget)\b[^|;\n]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b",
    r"\b(?:shutdown|reboot|halt|poweroff)\b(?:\s+-|\s+now|\s*$)",
    r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b",
    r"\bTRUNCATE\s+(?:TABLE\s+)?\w+",
    r"\bDELETE\s+FROM\s+[\w.\"`]+\s*(?:;|$)",
    r"\bgit\s+push\b[^\n]*(?:--force\b|\s-f\b)",
    r">\s*/dev/(?:sd[a-z]|nvme\d)",
    r"\bkill\s+-9\s+(?:-1|1)\b",
    r"\bformat\s+[a-z]:",
]
_DANGEROUS = re.compile("|".join(f"(?:{p})" for p in DANGEROUS_COMMAND_PATTERNS), re.I)

PRIVILEGE_PATTERNS = [
    r"\bsudo\s+\S",
    r"\bchmod\s+[ugo]*\+s\b",
    r"\bchown\s+root\b",
    r"\bGRANT\s+ALL\b",
    r"\bALTER\s+USER\b[^\n]*\b(?:SUPERUSER|ADMIN)\b",
    r"AdministratorAccess",
    r"\"Action\"\s*:\s*\"\*\"",
    r"\biam:\*",
    r"\b(?:add|grant)[_ -]?(?:admin|owner|role)\b",
    r"\bcluster-admin\b",
]
_PRIVILEGE = re.compile("|".join(f"(?:{p})" for p in PRIVILEGE_PATTERNS), re.I)

_URL = re.compile(r"\bhttps?://[^\s'\"<>]+", re.I)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")

# --- description / output scanning -------------------------------------------------

_HIDDEN_UNICODE = re.compile(
    "[\U000e0000-\U000e007f​-‏⁠-⁤﻿‪-‮⁦-⁩]"
)
_STRONG_DIRECTIVES = [
    ("ignore_previous", r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier|preceding|original)\s+(?:instructions?|prompts?|rules?|directions?)"),
    ("conceal_from_user", r"\b(?:do\s+not|don't|never)\s+(?:tell|inform|mention|notify|show|alert|reveal\s+to|let)\s+(?:this\s+to\s+)?(?:the\s+)?user\b|\bwithout\s+(?:telling|informing|notifying|alerting)\s+(?:the\s+)?user\b|\b(?:hide|conceal)\s+(?:this|it)\s+from\s+(?:the\s+)?user\b"),
    ("instruction_tag", r"<\s*/?\s*(?:important|system|instructions?|admin|secret|hidden)\s*>"),
]
_WEAK_DIRECTIVES = [
    ("precondition", r"\b(?:before|prior\s+to|after)\s+(?:using|calling|invoking|executing|running)\s+(?:this|any|the|each|every)?\s*['\"`]?\w*['\"`]?\s*(?:tool|function|command|operation)?"),
    ("mandatory_step", r"\b(?:mandatory|required|must|compulsory|obligatory)\b[^.]{0,80}\b(?:call|invoke|use|run|execute|send|read|write|forward|include)\b"),
    ("agent_directive", r"\b(?:you|the\s+(?:ai|assistant|agent|model|llm))\s+(?:must|should|need\s+to|are\s+required\s+to|have\s+to|shall)\b"),
    ("data_forwarding", r"\b(?:send|forward|upload|post|email|transmit|copy|exfiltrate)\b[^.]{0,80}\b(?:to|into)\b[^.]{0,40}(?:https?://|@|webhook|server|endpoint|address)"),
]
_STRONG_RE = [(n, re.compile(p, re.I)) for n, p in _STRONG_DIRECTIVES]
_WEAK_RE = [(n, re.compile(p, re.I)) for n, p in _WEAK_DIRECTIVES]
_ENCODED_BLOB = re.compile(r"[A-Za-z0-9+/]{160,}={0,2}")

OUTPUT_PATTERNS = [
    ("ignore_previous", _STRONG_RE[0][1]),
    ("conceal_from_user", _STRONG_RE[1][1]),
    ("instruction_tag", _STRONG_RE[2][1]),
    ("new_instructions", re.compile(r"\b(?:new|updated|additional|important)\s+(?:system\s+)?instructions?\s*(?::|for\s+(?:the\s+)?(?:ai|assistant|agent))", re.I)),
    ("role_marker", re.compile(r"(?:^|\n)\s*(?:system|assistant)\s*:\s", re.I)),
    ("tool_directive", re.compile(r"\b(?:call|invoke|use|run|execute)\s+(?:the\s+)?['\"`]?[a-z_][a-z0-9_]*['\"`]?\s+(?:tool|function)\b", re.I)),
    ("agent_directive", _WEAK_RE[2][1]),
    ("data_forwarding", _WEAK_RE[3][1]),
]
STRONG_OUTPUT = {"ignore_previous", "conceal_from_user", "instruction_tag", "new_instructions"}


def _domain_allowed(domain: str, allowlist: list[str]) -> bool:
    domain = domain.lower().rstrip(".")
    for entry in allowlist:
        entry = entry.lower().lstrip("*.")
        if domain == entry or domain.endswith("." + entry):
            return True
    return False


def call_signals(args: Any, allowlist: list[str]) -> dict[str, list[str]]:
    """Extract deterministic risk signals from tool-call arguments."""
    text = args if isinstance(args, str) else canonical_json(args)
    signals: dict[str, list[str]] = {}

    def add(kind: str, evidence: str) -> None:
        signals.setdefault(kind, [])
        if evidence not in signals[kind]:
            signals[kind].append(evidence[:120])

    for m in _SENSITIVE_PATH.finditer(text):
        add("sensitive_path", m.group(0).strip(" '\"="))
    for kind, _value in find_secrets(text):
        add("secret_in_args", kind)  # never echo any part of the secret
    for m in _DANGEROUS.finditer(text):
        add("dangerous_command", m.group(0))
    for m in _PRIVILEGE.finditer(text):
        add("privilege_change", m.group(0))
    for m in _URL.finditer(text):
        host = urlparse(m.group(0)).hostname or ""
        if host and not _domain_allowed(host, allowlist) and host not in ("localhost", "127.0.0.1"):
            add("external_egress", host)
    for m in _EMAIL.finditer(text):
        if not _domain_allowed(m.group(1), allowlist):
            add("external_egress", m.group(0))
    return signals


def description_signals(text: str, sibling_tools: list[str] | None = None) -> dict[str, list[str]]:
    signals: dict[str, list[str]] = {}
    if _HIDDEN_UNICODE.search(text):
        signals["hidden_unicode"] = ["invisible or tag characters present"]
    normalized = unicodedata.normalize("NFKC", _HIDDEN_UNICODE.sub("", text))
    for name, pattern in _STRONG_RE + _WEAK_RE:
        m = pattern.search(normalized)
        if m:
            signals.setdefault(name, []).append(m.group(0)[:120])
    for m in _SENSITIVE_PATH.finditer(normalized):
        signals.setdefault("sensitive_path", []).append(m.group(0).strip(" '\"="))
    if _ENCODED_BLOB.search(normalized):
        signals["encoded_payload"] = ["long base64-like blob"]
    for sibling in sibling_tools or []:
        if len(sibling) < 4:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(sibling)}(?![A-Za-z0-9_])", normalized):
            signals.setdefault("cross_tool_reference", []).append(sibling)
    return signals


STRONG_DESCRIPTION = {"hidden_unicode", "ignore_previous", "conceal_from_user", "instruction_tag"}


def output_signals(text: str) -> dict[str, list[str]]:
    signals: dict[str, list[str]] = {}
    if _HIDDEN_UNICODE.search(text):
        signals["hidden_unicode"] = ["invisible or tag characters present"]
    normalized = unicodedata.normalize("NFKC", _HIDDEN_UNICODE.sub("", text))
    for name, pattern in OUTPUT_PATTERNS:
        m = pattern.search(normalized)
        if m:
            signals.setdefault(name, []).append(m.group(0)[:120])
    return signals


# --- policy evaluation ---------------------------------------------------------------


@dataclass
class PolicyResult:
    action: Literal["allow", "ask", "deny", "guard"]
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, list[str]] = field(default_factory=dict)
    rule: str | None = None


class PolicyEngine:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self._rules = [(r, re.compile(r.pattern, re.I | re.S)) for r in config.rules]

    def evaluate_call(
        self, server_id: str, server: ServerConfig, tool_name: str, args: Any, intent: str | None = None
    ) -> PolicyResult:
        cfg = self.config
        allowlist = list(cfg.egress_allowlist) + list(server.egress_allowlist)
        signals = call_signals(args or {}, allowlist)
        if intent and "external_egress" in signals:
            # A destination the user named in their own request is not an escalation.
            named = intent.lower()
            remaining = [d for d in signals["external_egress"]
                         if d.lower() not in named and d.lower().removeprefix("www.") not in named]
            if remaining:
                signals["external_egress"] = remaining
            else:
                signals.pop("external_egress")
        tool_rule = server.tools.get(tool_name)

        if tool_rule and tool_rule.action == "deny":
            return PolicyResult("deny", [tool_rule.reason or f"tool '{tool_name}' is denied by policy"], signals, "tool_rule")

        canonical = canonical_json(args or {})
        for rule, pattern in self._rules:
            if not (fnmatch.fnmatch(server_id, rule.server) and fnmatch.fnmatch(tool_name, rule.tool)):
                continue
            if pattern.search(canonical):
                return PolicyResult(rule.action, [rule.reason or f"matched rule '{rule.name}'"], signals, rule.name)

        hard = []
        if cfg.block_sensitive_paths and "sensitive_path" in signals:
            hard.append(f"credential or key path in arguments: {', '.join(signals['sensitive_path'][:3])}")
        if cfg.block_secrets_in_args and "secret_in_args" in signals:
            hard.append(f"secret material in arguments: {', '.join(signals['secret_in_args'][:3])}")
        if cfg.block_dangerous_commands and "dangerous_command" in signals:
            hard.append(f"destructive command: {', '.join(signals['dangerous_command'][:3])}")
        if hard:
            return PolicyResult("deny", hard, signals, "hard_signal")

        soft = []
        if cfg.ask_on_external_egress and "external_egress" in signals:
            soft.append(f"sends data outside the allowlist: {', '.join(signals['external_egress'][:3])}")
        if cfg.ask_on_privilege_change and "privilege_change" in signals:
            soft.append(f"changes privileges: {', '.join(signals['privilege_change'][:3])}")
        if soft:
            return PolicyResult("ask", soft, signals, "soft_signal")

        if tool_rule and tool_rule.action in ("allow", "ask"):
            return PolicyResult(tool_rule.action, [tool_rule.reason or f"tool rule: {tool_rule.action}"], signals, "tool_rule")
        if server.default_action != "guard":
            return PolicyResult(server.default_action, [f"server default: {server.default_action}"], signals, "server_default")
        return PolicyResult("guard", [], signals)

    def evaluate_resource(self, uri: str) -> PolicyResult:
        signals = call_signals(uri, [])
        if self.config.block_sensitive_paths and "sensitive_path" in signals:
            return PolicyResult("deny", [f"credential or key path in resource URI: {uri}"], signals, "hard_signal")
        return PolicyResult("allow", [], signals)
