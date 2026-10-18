"""Human approval over MRTR (Multi Round-Trip Requests).

When a call needs approval the gateway answers with an ``InputRequiredResult``
containing a form elicitation and an HMAC-signed ``requestState``. The client
asks the user and retries the same call with ``inputResponses`` and the state.
The gateway holds no session: everything it needs is in the signed state,
which binds the principal, server, method, tool name, a digest of the
arguments and an expiry, as the spec requires for state that influences
authorization.

If the upstream server itself returns ``input_required``, its own
``requestState`` is carried inside the gateway's signed envelope (``inner``)
so both layers survive the round trip.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .protocol import sha256_hex

TOKEN_PREFIX = "mcpguard1."
APPROVAL_KEY = "mcpguard_approval"


class ApprovalError(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def args_digest(args: Any) -> str:
    return sha256_hex(args if args is not None else {})


@dataclass
class ApprovalState:
    principal: str
    server: str
    method: str
    name: str
    digest: str
    approved: bool
    inner: str | None
    expires_at: float
    nonce: str


class ApprovalSigner:
    def __init__(self, secret: bytes | str | None = None, ttl_s: int = 300):
        if isinstance(secret, str):
            secret = secret.encode("utf-8")
        # Without a configured secret, a random per-process key still works for
        # a single replica; multi-replica deployments must share MCPGUARD_STATE_SECRET.
        self._secret = secret or os.urandom(32)
        self.ttl_s = ttl_s
        self._used: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def issue(
        self,
        *,
        principal: str,
        server: str,
        method: str,
        name: str,
        digest: str,
        approved: bool,
        inner: str | None = None,
    ) -> str:
        payload = {
            "v": 1,
            "p": principal,
            "s": server,
            "m": method,
            "n": name,
            "d": digest,
            "a": approved,
            "i": inner,
            "exp": int(time.time()) + self.ttl_s,
            "nonce": secrets.token_urlsafe(9),
        }
        body = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        sig = _b64(hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest())
        return f"{TOKEN_PREFIX}{body}.{sig}"

    @staticmethod
    def is_gateway_token(value: Any) -> bool:
        return isinstance(value, str) and value.startswith(TOKEN_PREFIX)

    def verify(
        self, token: str, *, principal: str, server: str, method: str, name: str, digest: str
    ) -> ApprovalState:
        try:
            body, sig = token[len(TOKEN_PREFIX):].split(".", 1)
        except ValueError as exc:
            raise ApprovalError("malformed requestState") from exc
        expected = _b64(hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, sig):
            raise ApprovalError("requestState signature check failed")
        try:
            p = json.loads(_unb64(body))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ApprovalError("malformed requestState payload") from exc
        if p.get("exp", 0) < time.time():
            raise ApprovalError("requestState expired")
        checks = {"p": principal, "s": server, "m": method, "n": name, "d": digest}
        for key, value in checks.items():
            if p.get(key) != value:
                raise ApprovalError("requestState does not match this request")
        return ApprovalState(
            principal=p["p"], server=p["s"], method=p["m"], name=p["n"], digest=p["d"],
            approved=bool(p.get("a")), inner=p.get("i"), expires_at=p["exp"], nonce=p["nonce"],
        )

    def consume(self, nonce: str, expires_at: float) -> bool:
        """Best-effort single use within this replica. Returns False on replay."""
        now = time.time()
        with self._lock:
            while self._used and next(iter(self._used.values())) < now:
                self._used.popitem(last=False)
            if nonce in self._used:
                return False
            self._used[nonce] = expires_at
            return True


def approval_request(server: str, tool: str, args_preview: str, reasons: list[str]) -> dict:
    why = "; ".join(reasons) if reasons else "flagged by guard"
    message = (
        f"mcp-guard: approve call to '{tool}' on server '{server}'?\n"
        f"Reason: {why}\nArguments: {args_preview}"
    )
    return {
        APPROVAL_KEY: {
            "method": "elicitation/create",
            "params": {
                "mode": "form",
                "message": message,
                "requestedSchema": {
                    "type": "object",
                    "properties": {
                        "approve": {
                            "type": "boolean",
                            "title": "Allow this tool call?",
                            "default": False,
                        }
                    },
                    "required": ["approve"],
                },
            },
        }
    }


def approval_outcome(input_responses: Any) -> str:
    """Return 'approved', 'declined' or 'missing'."""
    if not isinstance(input_responses, dict) or APPROVAL_KEY not in input_responses:
        return "missing"
    resp = input_responses[APPROVAL_KEY]
    if not isinstance(resp, dict):
        return "missing"
    if resp.get("action") == "accept" and (resp.get("content") or {}).get("approve") is True:
        return "approved"
    return "declined"
