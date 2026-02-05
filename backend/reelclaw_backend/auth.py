from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import jwt


_BEARER_RE = re.compile(r"^Bearer\\s+(?P<t>.+)$", flags=re.IGNORECASE)


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    m = _BEARER_RE.match(authorization.strip())
    if not m:
        return None
    t = str(m.group("t") or "").strip()
    return t or None


@dataclass(frozen=True)
class JWTAuth:
    secret: str
    issuer: str = "reelclaw"
    ttl_seconds: int = 30 * 24 * 60 * 60  # 30 days

    def issue(self, *, user_id: str) -> str:
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": str(user_id),
            "iss": self.issuer,
            "iat": now,
            "exp": now + int(self.ttl_seconds),
        }
        return str(jwt.encode(payload, self.secret, algorithm="HS256"))

    def verify(self, token: str) -> dict[str, Any]:
        claims = jwt.decode(
            token,
            self.secret,
            algorithms=["HS256"],
            issuer=self.issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
        return claims if isinstance(claims, dict) else {}

