"""
This module defines the SQLAlchemy ORM model for storing sentiment analysis results.
Each Sentiment instance represents a record of a user's input and the corresponding
sentiment analysis outcome.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

# Base class for the model
base = declarative_base()

class Sentiment(base):
    __tablename__ = "sentiments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    user_text_input = Column(String, nullable=False)
    sentiment_label = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    created_time = Column(DateTime, default=datetime.now(timezone.utc))