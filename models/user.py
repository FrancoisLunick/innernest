from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

class User(Base):
    """
    User model is for storing users
    """
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key = True, autoincrement = True)
    email = Column(String(50), index = True, unique = True, nullable = False)
    username = Column(String(50), index = True, unique = True, nullable = False)
    created_at = Column(DateTime(timezone = True), server_default = func.now(), nullable = False)
    updated_at = Column(DateTime(timezone = True), server_default = func.now(), onupdate=func.now(), nullable = False)
    
    journals = relationship("JournalEntry", back_populates = "user")