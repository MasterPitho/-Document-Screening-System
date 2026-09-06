"""
Authentication helpers: PBKDF2 password hashing, opaque bearer tokens
(only SHA-256 hashes of tokens are stored), and FastAPI dependencies.

All persistence goes through the repositories on ``request.app.state``; the
API layer never touches SQLAlchemy sessions directly.
"""

from __future__ import annotations

import datetime
import hashlib
import secrets
from typing import Optional

from fastapi import HTTPException, Request

from app.config import Settings
from app.db.database import Database, utcnow_naive
from app.db.models import User
from app.models.schemas import UserOut


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"pbkdf2_sha256$100000${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        parts = stored.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_digest = parts[3]
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return secrets.compare_digest(digest.hex(), expected_digest)
    except (ValueError, IndexError):
        return False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def _is_expired(expires_at: Optional[datetime.datetime],
                now: Optional[datetime.datetime] = None) -> bool:
    """True if ``expires_at`` is in the past.

    PostgreSQL returns ``timestamptz`` values timezone-aware while SQLite
    returns them naive, so both are normalized to naive-UTC before comparing.
    """
    if expires_at is None:
        return False
    now = now if now is not None else utcnow_naive()
    if expires_at.tzinfo is not None:
        expires_at = expires_at.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return expires_at < now


def extract_optional_user(request: Request) -> Optional[User]:
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    token_hash = _hash_token(token)
    token_record, user = request.app.state.user_repo.find_by_token_hash(token_hash)
    if token_record is None or user is None:
        return None
    if _is_expired(token_record.expires_at):
        return None
    if not user.is_active:
        return None
    return user


def get_current_user(request: Request) -> User:
    user = extract_optional_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def as_user_out(user: User) -> UserOut:
    created = user.created_at
    return UserOut(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        created_at=created.isoformat() if created else "",
    )


def bootstrap_admin(database: Database, settings: Settings) -> None:
    if not settings.admin_username or not settings.admin_password or not settings.admin_email:
        return
    from app.db.repositories import UserRepository

    repo = UserRepository(database)
    if repo.get_by_username(settings.admin_username) is not None:
        return
    repo.create(
        username=settings.admin_username,
        email=settings.admin_email,
        full_name="System Admin",
        role="admin",
        password_hash=_hash_password(settings.admin_password),
    )
