"""Vetted view of each upstream server's tool catalog.

The gateway is stateless per request, so to judge a ``tools/call`` it needs
the tool's definition without relying on a prior ``tools/list`` from the same
client. It keeps a short-lived cache per replica, refreshed from upstream
using the ``ttlMs`` hint that 2026-07-28 list results carry.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .audit import AuditLog
from .config import Settings
from .guard import GuardEngine, Verdict
from .pinning import PinStore
from .protocol import gateway_meta
from .upstream import UpstreamClient


@dataclass
class Hidden:
    name: str
    reason: str
    kind: str  # rug_pull | unpinned | poisoned_description


@dataclass
class VettedCatalog:
    tools: dict[str, dict] = field(default_factory=dict)  # every tool upstream advertised
    hidden: dict[str, Hidden] = field(default_factory=dict)  # tools withheld from clients
    expires_at: float = 0.0


class ToolRegistry:
    def __init__(self, settings: Settings, upstream: UpstreamClient, guard: GuardEngine, pins: PinStore, audit: AuditLog):
        self.settings = settings
        self.upstream = upstream
        self.guard = guard
        self.pins = pins
        self.audit = audit
        self._catalogs: dict[str, VettedCatalog] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def vet(
        self, server_id: str, tools: list[dict], ttl_ms: float | None, principal: str, store: bool = True
    ) -> VettedCatalog:
        """Pin-check and scan a tool list. ``store`` caches it as the server's full catalog
        (pass False for a single page of a paginated list)."""
        server = self.settings.servers[server_id]
        names = [t.get("name", "") for t in tools]
        catalog = VettedCatalog()
        desc_jobs: list[tuple[dict, asyncio.Task[Verdict]]] = []

        for tool in tools:
            name = tool.get("name", "")
            catalog.tools[name] = tool
            pin = self.pins.check(server_id, tool, server.pinning.mode)
            if pin.status == "changed":
                catalog.hidden[name] = Hidden(name, "tool definition changed since it was pinned (possible rug pull)", "rug_pull")
                continue
            if pin.status == "unpinned":
                catalog.hidden[name] = Hidden(name, "tool is not in the reviewed lockfile (strict pinning)", "unpinned")
                continue
            if self.settings.description_scan.enabled:
                desc_jobs.append((tool, asyncio.ensure_future(self.guard.check_description(server_id, tool, names))))

        for tool, job in desc_jobs:
            verdict = await job
            name = tool.get("name", "")
            if verdict.decision == "deny" or (verdict.decision == "ask" and verdict.max_risk >= self.settings.description_scan.threshold):
                if self.settings.description_scan.action == "hide":
                    catalog.hidden[name] = Hidden(name, "; ".join(verdict.reasons) or "suspicious description", "poisoned_description")
                self.audit.record(
                    "description_flagged", principal=principal, server=server_id, tool=name,
                    decision=verdict.decision, source=verdict.source, reasons=verdict.reasons,
                    risks=verdict.risks or None, backend=verdict.backend,
                )

        for hidden in catalog.hidden.values():
            if hidden.kind != "poisoned_description":
                self.audit.record("tool_quarantined", principal=principal, server=server_id, tool=hidden.name,
                                  kind=hidden.kind, reasons=[hidden.reason])

        ttl_s = self.settings.gateway.list_cache_ttl_s
        if isinstance(ttl_ms, (int, float)) and ttl_ms >= 0:
            ttl_s = min(ttl_s, ttl_ms / 1000)
        catalog.expires_at = time.monotonic() + ttl_s
        if store:
            self._catalogs[server_id] = catalog
        return catalog

    async def get(self, server_id: str, principal: str) -> VettedCatalog:
        catalog = self._catalogs.get(server_id)
        if catalog and catalog.expires_at > time.monotonic():
            return catalog
        lock = self._locks.setdefault(server_id, asyncio.Lock())
        async with lock:
            catalog = self._catalogs.get(server_id)
            if catalog and catalog.expires_at > time.monotonic():
                return catalog
            tools, ttl_ms = await self.fetch_all(server_id)
            return await self.vet(server_id, tools, ttl_ms, principal)

    async def fetch_all(self, server_id: str) -> tuple[list[dict], float | None]:
        return await fetch_all_tools(self.upstream, server_id)

    def invalidate(self, server_id: str) -> None:
        self._catalogs.pop(server_id, None)


async def fetch_all_tools(upstream: UpstreamClient, server_id: str) -> tuple[list[dict], float | None]:
    """Fetch every page of tools/list. Returns (tools, ttlMs of the last page)."""
    tools: list[dict] = []
    cursor = None
    ttl_ms = None
    for _ in range(50):  # pagination guard
        params: dict = {"_meta": gateway_meta()}
        if cursor:
            params["cursor"] = cursor
        result = await upstream.call(server_id, "tools/list", params)
        tools.extend(result.get("tools") or [])
        ttl_ms = result.get("ttlMs", ttl_ms)
        cursor = result.get("nextCursor")
        if not cursor:
            break
    return tools, ttl_ms
