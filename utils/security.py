"""
Central place for auth/security helpers
"""

from __future__ import annotations

import os
from passlib.context import CryptContext
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JOSEError
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

pwd_context = CryptContext(schemes = ["bcrypt"], deprecated = "auto")

def hash_password(password: str) -> str:
    """
    Hash a plain text password for storage.

    Args:
        password (str): plain text password

    Returns:
        str: Hashed password
    """
    
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    
    """
    Verify a plain text password against a stored hash.
    """
    
    return pwd_context.verify(plain_password, hashed_password)
    
JWT_SECRET: str = os.getenv("JWT_SECRET_KEY", "dev-only-change-me")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
JWT_ISSUER: Optional[str] = os.getenv("JWT_ISSUER")
JWT_AUDIENCE: Optional[str] = os.getenv("JWT_AUDIENCE")
    

def create_access_token(data: Dict[str, Any]) -> str:
    """
    Create a signed JWT

    Args:
        data (Dict[str, Any]): Claims to embed in the token. Common claim is "sub" for user id.

    Returns:
        str: Encoded JWT string.
    """
    
    to_encode = data.copy()
    
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Standard registered claims
    to_encode.update({"exp": expire, "iat": now})
    
    # Optional hardening: issuer / audience if configured
    if JWT_ISSUER:
        to_encode.setdefault("iss", JWT_ISSUER)
    if JWT_AUDIENCE:
        to_encode.setdefault("aud", JWT_AUDIENCE)
    
    return jwt.encode(to_encode, JWT_SECRET, algorithm = JWT_ALGORITHM)
    

def decode_and_verify_token(token: str, *, issuer: Optional[str] = None, audience: Optional[str] = None) -> Dict[str, Any]:
    """
    Decode and validate a JWT

    Args:
        token (str): Raw JWT string
        issuer (Optional[str], optional): Expected issuer claim ("iss"). If None, falls back to JWT_ISSUER.
        audience (Optional[str], optional): Expected audience claim ("aud"). If None, falls back to JWT_AUDIENCE.

    Returns:
        Dict[str, Any]: Decoded JWT payload (claims).
    """
    
    issuer = issuer if issuer is not None else JWT_ISSUER
    audience = audience if audience is not None else JWT_AUDIENCE
    
    decode_kwargs: Dict[str, Any] = {}
    
    if issuer:
        decode_kwargs["issuer"] = issuer
    if audience:
        decode_kwargs["audience"] = audience
    
    
    return jwt.decode(
        token,
        JWT_SECRET,
        algorithms = [JWT_ALGORITHM],
        **decode_kwargs
        )

