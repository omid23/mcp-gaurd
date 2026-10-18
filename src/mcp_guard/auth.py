"""Who is calling. The principal is bound into approval state and the audit log."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .config import AuthConfig


class AuthError(Exception):
    pass


def parse_api_keys(raw: str) -> dict[str, str]:
    """``key1:alice,key2:bob`` -> {key1: alice, key2: bob}."""
    keys: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, principal = item.partition(":")
        keys[key.strip()] = principal.strip() or "api-key-user"
    return keys


class Authenticator:
    def __init__(self, cfg: AuthConfig):
        self.cfg = cfg
        self._keys = parse_api_keys(os.environ.get(cfg.api_keys_env, "")) if cfg.mode == "api_key" else {}

    def principal(self, headers: Mapping[str, str]) -> str:
        lower = {k.lower(): v for k, v in headers.items()}
        if self.cfg.mode == "none":
            return "anonymous"
        if self.cfg.mode == "header":
            value = lower.get(self.cfg.principal_header.lower())
            if not value:
                raise AuthError(f"missing {self.cfg.principal_header} header")
            return value
        auth = lower.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        principal = self._keys.get(token)
        if not principal:
            raise AuthError("invalid or missing API key")
        return principal
