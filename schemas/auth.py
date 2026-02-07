"""
Auth schemas define the data contracts for auth related actions.
"""

from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    """
    Schema used when a new user registers.
    
    This represents *untrusted input* coming directly from the client.
    It is validated by Pydantic before being passed to the service layer.
    """
    
    email: EmailStr     # Makes sure the email format is valid
    password: str       # Plain password (will be hashed later)
    username: str       # Public facing username
 
class UserPublic(BaseModel):
    """
    Schema used when returning a user object to the client.

    This schema is intentionally limited to non-sensitive fields.
    It is safe to expose publicly and is often used as a response_model.
    """
    
    id: int
    email: EmailStr
    
    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    """
    Schema used when a user attempts to log in.

    'identifier' allows flexibility (email or username),
    while keeping the schema minimal and explicit.
    """
    
    identifier: str     # Email or username
    password: str       # plain password (verified in the service layer)

