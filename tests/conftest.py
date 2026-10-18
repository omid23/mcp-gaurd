from __future__ import annotations

import httpx
import pytest

from mcp_guard.app import create_app
from mcp_guard.config import AuditConfig, PolicyConfig, ServerConfig, Settings, ToolRule
from mcp_guard.demo import DemoState, GatewayClient, build_demo_app


def demo_settings(**overrides) -> Settings:
    settings = Settings(
        servers={
            "workspace": ServerConfig(
                url="http://demo/workspace/mcp",
                tools={"list_dir": ToolRule(action="allow"), "deploy": ToolRule(action="allow")},
            ),
            "weather": ServerConfig(url="http://demo/weather/mcp"),
        },
        policy=PolicyConfig(egress_allowlist=["acme.example"]),
        audit=AuditConfig(jsonl_path=None),
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


class Harness:
    def __init__(self, settings: Settings | None = None, *, elicitation: bool = True, **gateway_kwargs):
        self.state = DemoState()
        self.demo_app = build_demo_app(self.state)
        self.upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.demo_app), base_url="http://demo")
        self.settings = settings or demo_settings()
        self.app = create_app(self.settings, upstream_client=self.upstream, state_secret="test-secret", **gateway_kwargs)
        self.gateway = self.app.state.gateway
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gw")
        self.client = GatewayClient("http://gw", client=self.http, elicitation=elicitation)

    async def call(self, server: str, tool: str, args: dict, **kw) -> dict:
        return await self.client.request(server, "tools/call", {"name": tool, "arguments": args}, **kw)

    async def rugpull(self) -> None:
        await self.upstream.post("/weather/admin/rugpull")

    async def aclose(self) -> None:
        await self.http.aclose()
        await self.upstream.aclose()


@pytest.fixture
async def harness():
    h = Harness()
    yield h
    await h.aclose()
