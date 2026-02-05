from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import jwt


APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"


@dataclass
class AppleJWKCache:
    jwks: dict[str, Any] | None = None
    fetched_at: float | None = None

    def get(self, *, max_age_s: float = 6 * 60 * 60) -> dict[str, Any]:
        now = time.time()
        if self.jwks and self.fetched_at and (now - self.fetched_at) < float(max_age_s):
            return self.jwks

        req = urllib.request.Request(
            APPLE_JWKS_URL,
            headers={"Accept": "application/json", "User-Agent": "reelclaw-backend/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        doc = json.loads(raw)
        if not isinstance(doc, dict) or "keys" not in doc:
            raise RuntimeError("Invalid Apple JWKS response")

        self.jwks = doc
        self.fetched_at = now
        return doc


_APPLE_JWK_CACHE = AppleJWKCache()


def verify_apple_identity_token(identity_token: str, *, audience: str) -> dict[str, Any]:
    """
    Verify Apple Sign-In identity token and return claims.
    """
    if not identity_token or not identity_token.strip():
        raise ValueError("Missing identity token")
    if not audience or not audience.strip():
        raise ValueError("Missing Apple audience (bundle id)")

    unverified_header = jwt.get_unverified_header(identity_token)
    kid = str(unverified_header.get("kid") or "").strip()
    if not kid:
        raise ValueError("Invalid identity token header (missing kid)")

    jwks = _APPLE_JWK_CACHE.get()
    keys = jwks.get("keys")
    if not isinstance(keys, list) or not keys:
        raise RuntimeError("Apple JWKS missing keys")

    jwk = None
    for k in keys:
        if isinstance(k, dict) and str(k.get("kid") or "") == kid:
            jwk = k
            break
    if not jwk:
        # Refresh once in case keys rotated.
        jwks = _APPLE_JWK_CACHE.get(max_age_s=0)
        keys = jwks.get("keys") if isinstance(jwks, dict) else None
        if isinstance(keys, list):
            for k in keys:
                if isinstance(k, dict) and str(k.get("kid") or "") == kid:
                    jwk = k
                    break
    if not jwk:
        raise RuntimeError("Apple public key not found for token kid")

    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
    claims = jwt.decode(
        identity_token,
        key=public_key,
        algorithms=["RS256"],
        audience=audience,
        issuer=APPLE_ISSUER,
        options={"require": ["exp", "iat", "sub"]},
    )
    return claims if isinstance(claims, dict) else {}

