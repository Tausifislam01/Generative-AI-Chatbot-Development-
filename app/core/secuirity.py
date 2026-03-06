from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from app.core.config import settings


password_hash = PasswordHash.recommeded()

def hash_password(password: str) -> str:
    return password_hash.hash(password)

def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)

def create_access_token(subject: str,
                        role: str,
                        expires_delta: Optional[timedelta] = None,
                        extra_claims: Optional[dict[str, Any]] = None,
                        ) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))

    payload: dict[str, Any] = {
        'sub': subject,
        'role':role,
        'iat':int(now.timestamp()),
        'exp': int(expire.timestamp()),
    }

    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.jwt_secret_key, algorithm = settings.jwt_algorithm,)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret_key, algorithms = [settings.jwt_algorithm],)


def get_token_subject(token_payload: dict[str, Any]) -> str:
    subject = token_payload.get('sub')
    if not isinstance(subject, str) or not subject.strip():
        raise InvalidTokenError("Token subject is missing or invalid")
    return subject

def get_token_role(token_payload: dict[str, Any]) -> str:
    role = token_payload.get('role')
    if not isinstance(role, str) or not role.strip():
        raise InvalidTokenError("Token role is missing or invalid")
    return role


    