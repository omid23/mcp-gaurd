"""Typed configuration loaded from YAML with ${ENV} / ${ENV:-default} expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Action = Literal["allow", "ask", "deny", "guard"]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_PATTERN.sub(repl, value)


def _expand(obj):
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


class ToolRule(BaseModel):
    """Per-tool override. ``guard`` sends the call to the decision model."""

    action: Action = "guard"
    reason: str | None = None


class PinningConfig(BaseModel):
    # off: no pinning; tofu: trust on first use, quarantine on change;
    # strict: only tools present in the lockfile are exposed.
    mode: Literal["off", "tofu", "strict"] = "tofu"


class ServerConfig(BaseModel):
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 30.0
    pinning: PinningConfig = Field(default_factory=PinningConfig)
    tools: dict[str, ToolRule] = Field(default_factory=dict)
    default_action: Action = "guard"
    egress_allowlist: list[str] = Field(default_factory=list)


class Thresholds(BaseModel):
    deny: float = 0.85
    ask: float = 0.5


class JevConfig(BaseModel):
    # Any server that speaks the /v1/systemone wire format works here:
    # TypeSafe's hosted API, or a local open-weight decision model server.
    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-1.13"
    api_key_env: str = "TYPESAFE_API_KEY"
    timeout_s: float = 3.0
    max_retries: int = 2
    input_cost_per_mtok: float = 0.042
    output_cost_per_mtok: float = 0.0


class LLMJudgeConfig(BaseModel):
    # Any OpenAI-compatible /chat/completions endpoint (OpenAI, Azure, vLLM, Ollama).
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-6-luna"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 10.0
    max_retries: int = 2
    input_cost_per_mtok: float = 0.10
    output_cost_per_mtok: float = 0.50


class VertexConfig(BaseModel):
    # Gemini on Google Cloud Vertex AI. Auth: an API key (Vertex AI express mode)
    # if api_key_env is set, otherwise Application Default Credentials (GKE
    # Workload Identity, a service account, or `gcloud auth application-default login`).
    model: str = "gemini-3.5-flash-lite"
    project: str | None = None     # default: $GOOGLE_CLOUD_PROJECT, then the ADC project
    location: str = "global"       # or a region such as us-central1 for data residency
    api_key_env: str = "GOOGLE_API_KEY"
    # Gemini's harm filters can block the attack text the judge has to read, so
    # they are relaxed by default; set to null to keep the project's defaults.
    safety_threshold: str | None = "BLOCK_NONE"
    timeout_s: float = 10.0
    max_retries: int = 2
    # Set from the Vertex AI pricing page for your model; 0 reports no cost.
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0


class ClassifierConfig(BaseModel):
    model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    injection_label: str = "INJECTION"
    device: int = -1


class DecisionConfig(BaseModel):
    backend: Literal["heuristic", "jev", "llm", "vertex", "classifier"] = "heuristic"
    fallback: Literal["heuristic", "none"] = "heuristic"
    thresholds: Thresholds = Field(default_factory=Thresholds)
    jev: JevConfig = Field(default_factory=JevConfig)
    llm: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)
    vertex: VertexConfig = Field(default_factory=VertexConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    cache_ttl_s: float = 60.0
    cache_size: int = 4096
    max_state_chars: int = 6000


class OutputScanConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.7
    # quarantine: withhold the whole result; block: withhold and mark isError;
    # annotate: prepend a warning; sanitize: remove only the injected passages,
    # falling back to quarantine if the remainder still fails the scan.
    action: Literal["quarantine", "block", "annotate", "sanitize"] = "quarantine"
    max_chars: int = 12000


class DescriptionScanConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.7
    action: Literal["hide", "annotate"] = "hide"


class ArgRule(BaseModel):
    name: str
    server: str = "*"
    tool: str = "*"
    pattern: str
    action: Literal["allow", "ask", "deny"]
    reason: str | None = None


class PolicyConfig(BaseModel):
    block_sensitive_paths: bool = True
    block_secrets_in_args: bool = True
    block_dangerous_commands: bool = True
    ask_on_external_egress: bool = True
    ask_on_privilege_change: bool = True
    egress_allowlist: list[str] = Field(default_factory=list)
    rules: list[ArgRule] = Field(default_factory=list)


class AuthConfig(BaseModel):
    # none: everyone is "anonymous"; api_key: Bearer keys mapped to principals;
    # header: trust a principal header set by an authenticating proxy.
    mode: Literal["none", "api_key", "header"] = "none"
    api_keys_env: str = "MCPGUARD_API_KEYS"
    principal_header: str = "x-forwarded-user"


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8800
    require_headers: bool = True
    protocol_versions: list[str] = Field(default_factory=lambda: ["2026-07-28"])
    state_secret_env: str = "MCPGUARD_STATE_SECRET"
    approval_ttl_s: int = 300
    fail_mode: Literal["ask", "deny"] = "ask"
    redact_before_model: bool = True
    list_cache_ttl_s: float = 300.0
    lockfile: str | None = None


class AuditConfig(BaseModel):
    jsonl_path: str | None = "audit.jsonl"
    include_arguments: bool = False


class TelemetryConfig(BaseModel):
    service_name: str = "mcp-guard"
    otlp_endpoint: str | None = None
    console: bool = False


class Settings(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    output_scan: OutputScanConfig = Field(default_factory=OutputScanConfig)
    description_scan: DescriptionScanConfig = Field(default_factory=DescriptionScanConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    servers: dict[str, ServerConfig] = Field(default_factory=dict)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


def load_settings(path: str | Path) -> Settings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Settings.model_validate(_expand(raw))
