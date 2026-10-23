"""A minimal MCP 2026-07-28 client and the scripted demo walkthrough."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable
from typing import Any

import httpx

from ..protocol import (
    META_CLIENT_CAPS,
    META_CLIENT_INFO,
    META_GUARD,
    META_INTENT,
    META_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    expected_name,
)

Approver = Callable[[str, dict], dict]


def auto_approver(key: str, request: dict) -> dict:
    """Simulates a user who accepts every form elicitation."""
    schema = (request.get("params") or {}).get("requestedSchema") or {}
    content = {k: True for k, v in (schema.get("properties") or {}).items() if v.get("type") == "boolean"}
    return {"action": "accept", "content": content}


class GatewayClient:
    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None, elicitation: bool = True,
                 api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=30)
        self.elicitation = elicitation
        self.api_key = api_key
        self._ids = itertools.count(1)

    def _meta(self, intent: str | None) -> dict:
        caps: dict = {"elicitation": {"form": {}}} if self.elicitation else {}
        meta = {
            META_PROTOCOL_VERSION: PROTOCOL_VERSION,
            META_CLIENT_INFO: {"name": "mcp-guard-demo-client", "version": "0.1.0"},
            META_CLIENT_CAPS: caps,
        }
        if intent:
            meta[META_INTENT] = intent
        return meta

    async def raw(self, server: str, method: str, params: dict, headers: dict | None = None) -> httpx.Response:
        body = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        hdrs = {"MCP-Protocol-Version": PROTOCOL_VERSION, "Mcp-Method": method,
                "Accept": "application/json, text/event-stream"}
        name = expected_name(method, params)
        if name is not None:
            hdrs["Mcp-Name"] = name
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        hdrs.update(headers or {})
        return await self._client.post(f"{self.base_url}/mcp/{server}", json=body, headers=hdrs)

    async def request(self, server: str, method: str, params: dict | None = None, *, intent: str | None = None,
                      approver: Approver | None = auto_approver, max_rounds: int = 3) -> dict:
        """Send a request, completing MRTR rounds with ``approver``. Returns the final JSON-RPC message."""
        params = dict(params or {})
        params["_meta"] = self._meta(intent)
        rounds: list[dict] = []
        for _ in range(max_rounds):
            resp = await self.raw(server, method, params)
            msg = resp.json()
            result = msg.get("result") or {}
            if result.get("resultType") != "input_required" or approver is None:
                msg["_rounds"] = rounds
                return msg
            rounds.append(result)
            responses = {k: approver(k, req) for k, req in (result.get("inputRequests") or {}).items()}
            params = {k: v for k, v in params.items() if k not in ("inputResponses", "requestState")}
            if responses:
                params["inputResponses"] = responses
            if result.get("requestState"):
                params["requestState"] = result["requestState"]
        msg["_rounds"] = rounds
        return msg

    async def aclose(self) -> None:
        await self._client.aclose()


def _summ(msg: dict) -> tuple[str, str]:
    if "error" in msg:
        return "error", f"{msg['error'].get('code')}: {msg['error'].get('message')}"
    result = msg.get("result") or {}
    guard = (result.get("_meta") or {}).get(META_GUARD) or {}
    text = " ".join(c.get("text", "") for c in result.get("content") or [] if isinstance(c, dict))
    if result.get("resultType") == "input_required":
        return "ask", "approval requested"
    status = "blocked" if result.get("isError") else guard.get("decision", "allow")
    output = guard.get("output", {})
    if output.get("decision") == "deny":
        status = "output sanitized" if output.get("action") == "sanitized" else "output withheld"
    if msg.get("_rounds"):
        status += f" (after {len(msg['_rounds'])} approval round{'s' if len(msg['_rounds']) > 1 else ''})"
    return status, " ".join(text.split())[:150]


async def run_demo(gateway_url: str, demo_url: str) -> list[tuple[str, str, str]]:
    client = GatewayClient(gateway_url)
    rows: list[tuple[str, str, str]] = []

    async def step(label: str, server: str, method: str, params: dict | None = None, **kw: Any) -> dict:
        msg = await client.request(server, method, params, **kw)
        if method == "tools/list":
            result = msg.get("result") or {}
            names = [t["name"] for t in result.get("tools", [])]
            hidden = ((result.get("_meta") or {}).get(META_GUARD) or {}).get("hiddenTools", [])
            detail = f"visible: {', '.join(names)}"
            if hidden:
                detail += " | hidden: " + ", ".join(f"{h['name']} ({h['kind']})" for h in hidden)
            rows.append((label, "listed", detail))
        else:
            status, detail = _summ(msg)
            rows.append((label, status, detail))
        return msg

    try:
        await step("1. list workspace tools", "workspace", "tools/list")
        await step("2. list weather tools", "weather", "tools/list")
        await step("3. read notes", "workspace", "tools/call",
                   {"name": "read_file", "arguments": {"path": "/workspace/notes.txt"}})
        await step("4. read SSH key", "workspace", "tools/call",
                   {"name": "read_file", "arguments": {"path": "~/.ssh/id_rsa"}})
        await step("5. rm -rf /", "workspace", "tools/call",
                   {"name": "run_shell", "arguments": {"command": "rm -rf / --no-preserve-root"}})
        await step("6. email outside the org", "workspace", "tools/call",
                   {"name": "send_email", "arguments": {"to": "partner@othercorp.example", "subject": "Q4 plan",
                                                        "body": "Sharing our Q4 plan."}})
        await step("7. read inbox (injected)", "workspace", "tools/call",
                   {"name": "read_file", "arguments": {"path": "/workspace/inbox.txt"}})
        await step("8. call poisoned tool", "weather", "tools/call",
                   {"name": "get_forecast", "arguments": {"city": "Toronto"}})
        await step("9. weather alerts (injected)", "weather", "tools/call",
                   {"name": "get_alerts", "arguments": {"city": "Toronto"}})
        await step("10. deploy (server MRTR)", "workspace", "tools/call",
                   {"name": "deploy", "arguments": {"environment": "staging"}})
        async with httpx.AsyncClient() as h:
            await h.post(f"{demo_url.rstrip('/')}/weather/admin/rugpull")
        await step("11. rug pull, list again", "weather", "tools/list")
        await step("12. call rug-pulled tool", "weather", "tools/call",
                   {"name": "get_time", "arguments": {"city": "Toronto"}})
    finally:
        await client.aclose()
    return rows


def print_rows(rows: list[tuple[str, str, str]]) -> None:
    width = max(len(r[0]) for r in rows) + 2
    for label, status, detail in rows:
        print(f"{label:<{width}}{status:<33}{detail}")


def main(gateway_url: str = "http://localhost:8800", demo_url: str = "http://localhost:9100") -> None:
    rows = asyncio.run(run_demo(gateway_url, demo_url))
    print_rows(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
