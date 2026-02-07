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
