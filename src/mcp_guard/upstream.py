"""Forwarding to upstream MCP servers over Streamable HTTP."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import ServerConfig, expand_env
from .protocol import H_METHOD, H_NAME, H_PARAM_PREFIX, H_PROTOCOL, PROTOCOL_VERSION, expected_name, get_params

# Client headers that are safe to pass upstream. The client's Authorization
# header is deliberately NOT forwarded: passing a client token through to a
# downstream API is the "token passthrough" anti-pattern the MCP authorization
# spec forbids. Upstream credentials come from gateway config instead.
FORWARD_HEADERS = {H_PROTOCOL, H_METHOD, H_NAME, "traceparent", "tracestate", "baggage"}


class UpstreamError(Exception):
    pass


@dataclass
class UpstreamResponse:
    status: int
    message: dict | None
    notifications: list[dict] = field(default_factory=list)


def parse_sse(text: str) -> list[dict]:
    messages: list[dict] = []
    data_lines: list[str] = []

    def flush() -> None:
        if data_lines:
            payload = "\n".join(data_lines)
            data_lines.clear()
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                return
            if isinstance(obj, dict):
                messages.append(obj)

    for line in text.splitlines():
        if not line.strip():
            flush()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    flush()
    return messages


class UpstreamClient:
    def __init__(self, servers: Mapping[str, ServerConfig], client: httpx.AsyncClient | None = None):
        self.servers = servers
        self._own = client is None
        self._client = client or httpx.AsyncClient()

    def _headers(self, server: ServerConfig, body: dict, client_headers: Mapping[str, str]) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        for key, value in client_headers.items():
            low = key.lower()
            if low in FORWARD_HEADERS or low.startswith(H_PARAM_PREFIX):
                headers[key] = value
        method = body.get("method")
        if isinstance(method, str):
            headers.setdefault("Mcp-Method", method)
            name = expected_name(method, get_params(body))
            if name is not None:
                headers["Mcp-Name"] = name
        headers.setdefault("MCP-Protocol-Version", PROTOCOL_VERSION)
        for key, value in server.headers.items():
            headers[key] = expand_env(value)
        return headers

    async def send(self, server_id: str, body: dict, client_headers: Mapping[str, str] | None = None) -> UpstreamResponse:
        server = self.servers[server_id]
        headers = self._headers(server, body, client_headers or {})
        try:
            resp = await self._client.post(server.url, json=body, headers=headers, timeout=server.timeout_s)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"upstream '{server_id}' unreachable: {type(exc).__name__}: {exc}") from exc
        if resp.status_code == 202 or not resp.content:
            return UpstreamResponse(resp.status_code, None)
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            messages = parse_sse(resp.text)
        else:
            try:
                messages = [resp.json()]
            except ValueError as exc:
                raise UpstreamError(f"upstream '{server_id}' returned non-JSON (HTTP {resp.status_code})") from exc
        req_id = body.get("id")
        final = None
        notes: list[dict] = []
        for msg in messages:
            if msg.get("id") == req_id and ("result" in msg or "error" in msg):
                final = msg
            else:
                notes.append(msg)
        if final is None and resp.status_code >= 400:
            raise UpstreamError(f"upstream '{server_id}' returned HTTP {resp.status_code}")
        return UpstreamResponse(resp.status_code, final, notes)

    async def stream(self, server_id: str, body: dict, client_headers: Mapping[str, str]) -> AsyncIterator[bytes]:
        """Pass a long-lived response stream through untouched (subscriptions/listen)."""
        server = self.servers[server_id]
        headers = self._headers(server, body, client_headers)
        async with self._client.stream("POST", server.url, json=body, headers=headers, timeout=None) as resp:
            async for chunk in resp.aiter_raw():
                yield chunk

    async def call(self, server_id: str, method: str, params: dict[str, Any], req_id: Any = "mcp-guard") -> dict:
        resp = await self.send(server_id, {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        if resp.message is None:
            raise UpstreamError(f"upstream '{server_id}' returned no response to {method}")
        if "error" in resp.message:
            err = resp.message["error"]
            raise UpstreamError(f"upstream '{server_id}' {method} error {err.get('code')}: {err.get('message')}")
        return resp.message.get("result") or {}

    async def aclose(self) -> None:
        if self._own:
            await self._client.aclose()
