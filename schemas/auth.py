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
 
