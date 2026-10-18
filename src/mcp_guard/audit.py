"""Append-only JSONL audit trail of every guard decision."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AuditConfig
from .protocol import sha256_hex
from .telemetry import current_trace_id


class AuditLog:
    def __init__(self, cfg: AuditConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._path = Path(cfg.jsonl_path) if cfg.jsonl_path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self.recent: list[dict] = []  # small in-memory tail, used by tests and the demo

    def record(self, event: str, *, principal: str, server: str, arguments: Any = None, **fields: Any) -> dict:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "principal": principal,
            "server": server,
        }
        trace_id = current_trace_id()
        if trace_id:
            entry["trace_id"] = trace_id
        if arguments is not None:
            entry["args_sha256"] = sha256_hex(arguments)
            if self.cfg.include_arguments:
                entry["arguments"] = arguments
        entry.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            self.recent.append(entry)
            del self.recent[:-500]
            if self._path:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        return entry
