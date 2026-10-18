"""MCP 2026-07-28 wire-level helpers.

The gateway works at the JSON-RPC level rather than through an SDK so it can
inspect and rewrite every message. Everything here follows the stateless
2026-07-28 revision: no initialize handshake, per-request ``_meta``, required
``Mcp-Method`` / ``Mcp-Name`` headers and Multi Round-Trip Requests (MRTR).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = "2026-07-28"

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

# Custom _meta keys use a reverse-DNS prefix, as the spec requires for
# implementation-specific metadata.
META_INTENT = "dev.mcpguard/userIntent"
META_GUARD = "dev.mcpguard/decision"

H_PROTOCOL = "mcp-protocol-version"
H_METHOD = "mcp-method"
H_NAME = "mcp-name"
H_PARAM_PREFIX = "mcp-param-"

# JSON-RPC / MCP error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
HEADER_MISMATCH = -32020
UNSUPPORTED_PROTOCOL_VERSION = -32022
# -32000..-32019 is implementation-defined; the gateway uses -32010.
GUARD_ERROR = -32010

# Methods whose Mcp-Name header must mirror a field of params.
NAMED_METHODS: dict[str, str] = {
    "tools/call": "name",
    "prompts/get": "name",
    "resources/read": "uri",
}
# Methods that may answer with an InputRequiredResult.
MRTR_METHODS = frozenset(NAMED_METHODS)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: Any) -> str:
    data = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def jsonrpc_result(req_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def get_params(body: Mapping[str, Any]) -> dict:
    params = body.get("params")
    return params if isinstance(params, dict) else {}


def get_meta(params: Mapping[str, Any]) -> dict:
    meta = params.get("_meta")
    return meta if isinstance(meta, dict) else {}


def expected_name(method: str, params: Mapping[str, Any]) -> str | None:
    field = NAMED_METHODS.get(method)
    if field is None:
        return None
    value = params.get(field)
    return value if isinstance(value, str) else None


@dataclass
class HeaderProblem:
    code: int
    message: str


def check_headers(
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    supported_versions: list[str],
    require: bool = True,
) -> HeaderProblem | None:
    """Validate the Streamable HTTP headers against the JSON-RPC body.

    A gateway that routes or authorizes on headers must refuse requests whose
    headers disagree with the body; otherwise a client could show the policy
    layer one tool name and execute another.
    """
    method = body.get("method")
    params = get_params(body)
    meta = get_meta(params)
    lower = {k.lower(): v for k, v in headers.items()}

    h_method = lower.get(H_METHOD)
    if h_method is None:
        if require:
            return HeaderProblem(HEADER_MISMATCH, "Missing required Mcp-Method header")
    elif h_method != method:
        return HeaderProblem(
            HEADER_MISMATCH, f"Mcp-Method header '{h_method}' does not match body method '{method}'"
        )

    name = expected_name(str(method), params)
    if name is not None:
        h_name = lower.get(H_NAME)
        if h_name is None:
            if require:
                return HeaderProblem(HEADER_MISMATCH, "Missing required Mcp-Name header")
        elif h_name != name:
            return HeaderProblem(
                HEADER_MISMATCH, f"Mcp-Name header '{h_name}' does not match body value '{name}'"
            )

    h_version = lower.get(H_PROTOCOL)
    m_version = meta.get(META_PROTOCOL_VERSION)
    if h_version and m_version and h_version != m_version:
        return HeaderProblem(
            HEADER_MISMATCH,
            f"MCP-Protocol-Version header '{h_version}' does not match _meta '{m_version}'",
        )
    version = m_version or h_version
    if version and version not in supported_versions:
        return HeaderProblem(
            UNSUPPORTED_PROTOCOL_VERSION,
            f"Unsupported protocol version '{version}'; supported: {', '.join(supported_versions)}",
        )
    return None


def client_supports_elicitation(meta: Mapping[str, Any]) -> bool:
    caps = meta.get(META_CLIENT_CAPS)
    return isinstance(caps, dict) and "elicitation" in caps


def tool_result(text: str, *, is_error: bool, guard_meta: dict | None = None) -> dict:
    result: dict[str, Any] = {
        "resultType": "complete",
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }
    if guard_meta:
        result["_meta"] = {META_GUARD: guard_meta}
    return result


def input_required(input_requests: dict | None, request_state: str | None) -> dict:
    result: dict[str, Any] = {"resultType": "input_required"}
    if input_requests:
        result["inputRequests"] = input_requests
    if request_state:
        result["requestState"] = request_state
    return result


def add_guard_meta(result: dict, guard_meta: dict) -> dict:
    meta = result.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        result["_meta"] = meta
    meta[META_GUARD] = guard_meta
    return result


def extract_text(result: Mapping[str, Any]) -> str:
    """Collect model-visible text from a tools/call, resources/read or prompts/get result."""
    parts: list[str] = []

    def from_content(item: Any) -> None:
        if not isinstance(item, dict):
            return
        if isinstance(item.get("text"), str):
            parts.append(item["text"])
        resource = item.get("resource")
        if isinstance(resource, dict) and isinstance(resource.get("text"), str):
            parts.append(resource["text"])

    for item in result.get("content") or []:
        from_content(item)
    for item in result.get("contents") or []:  # resources/read
        from_content(item)
    for msg in result.get("messages") or []:  # prompts/get
        if isinstance(msg, dict):
            from_content(msg.get("content"))
    structured = result.get("structuredContent")
    if structured is not None:
        parts.append(canonical_json(structured))
    return "\n".join(parts)


def gateway_meta(client_name: str = "mcp-guard", version: str = "0.1.0") -> dict:
    """_meta the gateway attaches to requests it originates (e.g. tools/list refreshes)."""
    return {
        META_PROTOCOL_VERSION: PROTOCOL_VERSION,
        META_CLIENT_INFO: {"name": client_name, "version": version},
        META_CLIENT_CAPS: {},
    }
