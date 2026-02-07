"""
Central place for auth/security helpers
"""

from __future__ import annotations

import os
from passlib.context import CryptContext
from jose import jwt
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
    expire = datetime.now(timezone.utc) + timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    return jwt.encode(to_encode, JWT_SECRET, algorithm = JWT_ALGORITHM)
    
