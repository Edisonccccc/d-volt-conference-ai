"""Password hashing, JWT signing/decoding, and FastAPI auth dependencies.

Self-contained so the auth surface stays in one place. Nothing imports
storage from here directly — the dependency calls back into storage to
fetch the user. That keeps storage agnostic to auth concerns.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt


log = logging.getLogger("conference-ai.auth")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JWT_ALG = "HS256"
JWT_EXPIRY_DAYS = int(os.getenv("JWT_EXPIRY_DAYS", "7"))


def _resolve_secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        # Generate a per-process secret in dev so login still works, but log
        # loudly so prod misconfig doesn't slip through.
        secret = secrets.token_hex(32)
        log.warning(
            "JWT_SECRET env var not set; generated a one-shot dev secret. "
            "Set JWT_SECRET in production so tokens survive restarts."
        )
    return secret


JWT_SECRET = _resolve_secret()


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_hasher = PasswordHasher()


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, plain)
    except (VerifyMismatchError, InvalidHashError):
        return False


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def make_jwt(user_id: str, *, expiry_days: int = JWT_EXPIRY_DAYS) -> str:
    """Sign a JWT with `sub: user_id` and an expiry."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + expiry_days * 24 * 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_jwt(token: str) -> Optional[str]:
    """Return the `sub` (user_id) from a valid token, else None."""
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return data.get("sub")
    except JWTError as exc:
        log.debug("JWT decode failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

# auto_error=False so we can give a friendlier 401 message than the default.
_bearer = HTTPBearer(auto_error=False)


def _credentials_dep(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header (use 'Bearer <token>').",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = decode_jwt(creds.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


def current_user_id(user_id: str = Depends(_credentials_dep)) -> str:
    """Lightweight dependency: just the user_id from the token.
    Use this when you don't need the full user record."""
    return user_id


def make_current_user_dep(get_user_by_id):
    """Build a dependency that returns the full User record.

    `get_user_by_id` is injected at app startup (to avoid importing
    storage.py here, which would create a circular dep).
    """

    def _dep(user_id: str = Depends(_credentials_dep)):
        user = get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User no longer exists.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    return _dep


def require_manager(user) -> None:
    """Raise 403 if the user isn't a manager. Call at the top of admin handlers."""
    if getattr(user, "role", None) != "manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is manager-only.",
        )


# ---------------------------------------------------------------------------
# Email allowlist
# ---------------------------------------------------------------------------

def email_allowed(email: str) -> bool:
    """Return True if the email's domain is in EMAIL_ALLOWLIST.

    EMAIL_ALLOWLIST is a comma-separated list of domains, e.g. 'd-volt.co'.
    Empty / unset means anyone can register (suitable for dev only).
    """
    raw = os.getenv("EMAIL_ALLOWLIST", "").strip()
    if not raw:
        return True
    allowed = [d.strip().lower() for d in raw.split(",") if d.strip()]
    domain = (email.split("@", 1)[-1] if "@" in email else "").lower()
    return domain in allowed
