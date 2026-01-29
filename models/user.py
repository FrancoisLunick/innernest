from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

class User(Base):
    """
    User model is for storing users
    """
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key = True, autoincrement = True)
    email = Column(String(50), nullable = False)
    username = Column(String(50), nullable = False)
    created_at = Column(DateTime, default=datetime.now)
    update_by = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    journals = relationship("JournalEntry", back_populates = "user")