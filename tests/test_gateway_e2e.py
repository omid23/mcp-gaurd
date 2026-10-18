"""End-to-end: client -> gateway -> demo MCP servers, all in process."""

from __future__ import annotations

import httpx

from mcp_guard.approvals import APPROVAL_KEY
from mcp_guard.protocol import HEADER_MISMATCH, META_GUARD, UNSUPPORTED_PROTOCOL_VERSION
from mcp_guard.upstream import UpstreamClient

from .conftest import Harness


def guard_meta(msg: dict) -> dict:
    return ((msg.get("result") or {}).get("_meta") or {}).get(META_GUARD) or {}


def text_of(msg: dict) -> str:
    return " ".join(c.get("text", "") for c in (msg.get("result") or {}).get("content", []))


async def test_discover_passes_through(harness: Harness):
    msg = await harness.client.request("workspace", "server/discover")
    assert msg["result"]["supportedVersions"] == ["2026-07-28"]


async def test_poisoned_tool_hidden_from_list(harness: Harness):
    msg = await harness.client.request("weather", "tools/list")
    names = [t["name"] for t in msg["result"]["tools"]]
    assert "get_forecast" not in names
    assert {"get_alerts", "get_time"} <= set(names)
    hidden = guard_meta(msg)["hiddenTools"]
    assert hidden[0]["name"] == "get_forecast" and hidden[0]["kind"] == "poisoned_description"


async def test_hidden_tool_cannot_be_called(harness: Harness):
    msg = await harness.call("weather", "get_forecast", {"city": "Toronto"})
    assert msg["result"]["isError"] is True
    assert "quarantined" in text_of(msg)


async def test_benign_call_allowed(harness: Harness):
    msg = await harness.call("workspace", "read_file", {"path": "/workspace/notes.txt"})
    assert msg["result"]["isError"] is False
    assert "Q4 plan" in text_of(msg)
    assert guard_meta(msg)["decision"] == "allow"


async def test_credential_path_denied_before_upstream(harness: Harness):
    msg = await harness.call("workspace", "read_file", {"path": "~/.ssh/id_rsa"})
    assert msg["result"]["isError"] is True
    assert guard_meta(msg)["source"] == "policy"


async def test_destructive_command_denied_and_never_executed(harness: Harness):
    msg = await harness.call("workspace", "run_shell", {"command": "rm -rf / --no-preserve-root"})
    assert msg["result"]["isError"] is True
    assert harness.state.shell_log == []


async def test_external_email_requires_and_honors_approval(harness: Harness):
    args = {"to": "partner@othercorp.example", "subject": "Q4", "body": "plan"}
    pending = await harness.call("workspace", "send_email", args, approver=None)
    result = pending["result"]
    assert result["resultType"] == "input_required"
    assert APPROVAL_KEY in result["inputRequests"]
    assert result["inputRequests"][APPROVAL_KEY]["method"] == "elicitation/create"
    assert harness.state.outbox == []

    done = await harness.call("workspace", "send_email", args)  # auto-approves
    assert done["result"]["isError"] is False
    assert len(done["_rounds"]) == 1
    assert harness.state.outbox[0]["to"] == "partner@othercorp.example"


async def test_internal_email_needs_no_approval(harness: Harness):
    msg = await harness.call("workspace", "send_email", {"to": "sam@acme.example", "subject": "hi", "body": "lunch?"},
                             approver=None)
    assert msg["result"]["resultType"] == "complete"


async def test_declined_approval_blocks(harness: Harness):
    def decline(key, req):
        return {"action": "decline"}

    msg = await harness.call("workspace", "send_email", {"to": "x@othercorp.example", "subject": "s", "body": "b"},
                             approver=decline)
    assert msg["result"]["isError"] is True
    assert "declined" in text_of(msg)
    assert harness.state.outbox == []


async def test_approval_state_bound_to_arguments(harness: Harness):
    args = {"to": "partner@othercorp.example", "subject": "Q4", "body": "plan"}
    pending = await harness.call("workspace", "send_email", args, approver=None)
    token = pending["result"]["requestState"]
    swapped = {**args, "to": "attacker@evil.example"}
    params = {"name": "send_email", "arguments": swapped, "requestState": token,
              "inputResponses": {APPROVAL_KEY: {"action": "accept", "content": {"approve": True}}},
              "_meta": harness.client._meta(None)}
    resp = await harness.client.raw("workspace", "tools/call", params)
    msg = resp.json()
    assert msg["error"]["code"] == -32602
    assert harness.state.outbox == []


async def test_tampered_state_rejected(harness: Harness):
    args = {"to": "partner@othercorp.example", "subject": "Q4", "body": "plan"}
    pending = await harness.call("workspace", "send_email", args, approver=None)
    token = pending["result"]["requestState"]
    params = {"name": "send_email", "arguments": args, "requestState": token[:-2] + "AA",
              "inputResponses": {APPROVAL_KEY: {"action": "accept", "content": {"approve": True}}},
              "_meta": harness.client._meta(None)}
    msg = (await harness.client.raw("workspace", "tools/call", params)).json()
    assert "signature" in msg["error"]["message"]


async def test_approval_cannot_be_replayed(harness: Harness):
    args = {"to": "partner@othercorp.example", "subject": "Q4", "body": "plan"}
    pending = await harness.call("workspace", "send_email", args, approver=None)
    params = {"name": "send_email", "arguments": args, "requestState": pending["result"]["requestState"],
              "inputResponses": {APPROVAL_KEY: {"action": "accept", "content": {"approve": True}}},
              "_meta": harness.client._meta(None)}
    first = (await harness.client.raw("workspace", "tools/call", params)).json()
    second = (await harness.client.raw("workspace", "tools/call", params)).json()
    assert first["result"]["isError"] is False
    assert "already used" in second["error"]["message"]
    assert len(harness.state.outbox) == 1


async def test_client_without_elicitation_gets_deny_instead_of_ask():
    h = Harness(elicitation=False)
    try:
        msg = await h.call("workspace", "send_email", {"to": "p@othercorp.example", "subject": "s", "body": "b"})
        assert msg["result"]["isError"] is True
        assert "elicitation" in text_of(msg)
    finally:
        await h.aclose()


async def test_upstream_mrtr_state_survives_through_gateway(harness: Harness):
    msg = await harness.call("workspace", "deploy", {"environment": "staging"})
    assert "deployed build 1.4.2 to staging" in text_of(msg)
    assert len(msg["_rounds"]) == 1
    # The client only ever saw the gateway's wrapper, never the upstream state.
    assert msg["_rounds"][0]["requestState"].startswith("mcpguard1.")


async def test_injected_output_is_withheld(harness: Harness):
    msg = await harness.call("workspace", "read_file", {"path": "/workspace/inbox.txt"})
    assert "withheld" in text_of(msg)
    assert "billing@supplier-payments.example" not in text_of(msg)
    assert guard_meta(msg)["output"]["decision"] == "deny"


async def test_weak_signal_injection_withheld_by_model_layer(harness: Harness):
    msg = await harness.call("weather", "get_alerts", {"city": "Toronto"})
    assert "withheld" in text_of(msg)
    assert guard_meta(msg)["output"]["source"] == "model"


async def test_rug_pull_quarantines_changed_tool(harness: Harness):
    first = await harness.client.request("weather", "tools/list")
    assert "get_time" in [t["name"] for t in first["result"]["tools"]]
    await harness.rugpull()
    second = await harness.client.request("weather", "tools/list")
    assert "get_time" not in [t["name"] for t in second["result"]["tools"]]
    kinds = {h["name"]: h["kind"] for h in guard_meta(second)["hiddenTools"]}
    assert kinds["get_time"] == "rug_pull"
    harness.gateway.registry.invalidate("weather")
    call = await harness.call("weather", "get_time", {"city": "Toronto"})
    assert call["result"]["isError"] is True


async def test_unknown_tool_rejected(harness: Harness):
    msg = await harness.call("workspace", "format_disk", {})
    assert msg["error"]["code"] == -32602


async def test_header_body_mismatch_rejected(harness: Harness):
    params = {"name": "read_file", "arguments": {"path": "/workspace/notes.txt"}, "_meta": harness.client._meta(None)}
    resp = await harness.client.raw("workspace", "tools/call", params, headers={"Mcp-Name": "send_email"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == HEADER_MISMATCH


async def test_missing_method_header_rejected(harness: Harness):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp = await harness.http.post("/mcp/workspace", json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == HEADER_MISMATCH


async def test_unsupported_version_rejected(harness: Harness):
    params = {"_meta": {**harness.client._meta(None), "io.modelcontextprotocol/protocolVersion": "2024-11-05"}}
    resp = await harness.client.raw("workspace", "tools/list", params, headers={"MCP-Protocol-Version": "2024-11-05"})
    assert resp.json()["error"]["code"] == UNSUPPORTED_PROTOCOL_VERSION


async def test_batch_rejected(harness: Harness):
    resp = await harness.http.post("/mcp/workspace", json=[{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
    assert resp.status_code == 400


async def test_notification_accepted(harness: Harness):
    body = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 9}}
    resp = await harness.http.post("/mcp/workspace", json=body, headers={"Mcp-Method": "notifications/cancelled"})
    assert resp.status_code == 202


async def test_audit_records_decisions(harness: Harness):
    await harness.call("workspace", "read_file", {"path": "~/.ssh/id_rsa"})
    entry = harness.gateway.audit.recent[-1]
    assert entry["event"] == "tool_call" and entry["decision"] == "deny"
    assert "arguments" not in entry and len(entry["args_sha256"]) == 64


async def test_client_authorization_not_forwarded_upstream(harness: Harness):
    seen = {}

    async def capture(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})

    client = UpstreamClient(harness.settings.servers, httpx.AsyncClient(transport=httpx.MockTransport(capture)))
    await client.send("workspace", {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                      {"Authorization": "Bearer user-token", "Mcp-Method": "tools/list"})
    assert "authorization" not in {k.lower() for k in seen}
    assert seen["mcp-method"] == "tools/list"
