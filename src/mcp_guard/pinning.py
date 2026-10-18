"""Tool definition pinning: detects "rug pulls", where a server changes a tool
after it was reviewed or first seen.

``tofu`` pins on first sight, per replica. ``strict`` exposes only tools whose
fingerprint is in a reviewed lockfile, which is the mode to run with several
replicas, since every replica loads the same pins.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .protocol import sha256_hex

FINGERPRINT_FIELDS = ("name", "title", "description", "inputSchema", "outputSchema", "annotations")


def tool_fingerprint(tool: dict) -> str:
    return sha256_hex({k: tool.get(k) for k in FINGERPRINT_FIELDS})


@dataclass
class PinCheck:
    status: Literal["pinned", "new", "changed", "unpinned", "off"]
    fingerprint: str
    expected: str | None = None

    @property
    def quarantined(self) -> bool:
        return self.status in ("changed", "unpinned")


class PinStore:
    def __init__(self, lockfile: str | Path | None = None):
        self._pins: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        if lockfile and Path(lockfile).exists():
            self.load(lockfile)

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for server, tools in data.get("servers", {}).items():
            self._pins[server] = {name: entry["sha256"] for name, entry in tools.items()}

    def check(self, server: str, tool: dict, mode: str) -> PinCheck:
        fp = tool_fingerprint(tool)
        if mode == "off":
            return PinCheck("off", fp)
        name = tool.get("name", "")
        with self._lock:
            pinned = self._pins.get(server, {}).get(name)
            if pinned is None:
                if mode == "strict":
                    return PinCheck("unpinned", fp)
                self._pins.setdefault(server, {})[name] = fp
                return PinCheck("new", fp)
        if pinned == fp:
            return PinCheck("pinned", fp, pinned)
        return PinCheck("changed", fp, pinned)

    def repin(self, server: str, tool: dict) -> None:
        with self._lock:
            self._pins.setdefault(server, {})[tool.get("name", "")] = tool_fingerprint(tool)

    def snapshot(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return {s: dict(t) for s, t in self._pins.items()}


def write_lockfile(path: str | Path, server_tools: dict[str, list[dict]]) -> dict:
    """Write a reviewable lockfile: fingerprints plus the description that was reviewed."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data: dict = {"version": 1, "generated_at": now, "servers": {}}
    existing: dict = {}
    if Path(path).exists():
        existing = json.loads(Path(path).read_text(encoding="utf-8")).get("servers", {})
    data["servers"].update(existing)
    for server, tools in server_tools.items():
        data["servers"][server] = {
            t["name"]: {
                "sha256": tool_fingerprint(t),
                "description": (t.get("description") or "")[:500],
                "pinned_at": now,
            }
            for t in sorted(tools, key=lambda t: t.get("name", ""))
        }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
