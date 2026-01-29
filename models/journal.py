from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from models.user import User

class JournalEntry:
    """
    JournalEntry model is for storing user written journal entries.
    """
    
    __tablename__ = "journal_entries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    mood = Column(String(50), nullable=True)
    tags = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    created_by = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    user = relationship("User", back_populates="journal_entries")
    
    def __repr__(self) -> str:
        return f"<JournalEntry id={self.id} user_id={self.user_id} mood={self.mood}>"