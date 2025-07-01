"""
This module defines the SQLAlchemy ORM model for storing sentiment analysis results.
Each Sentiment instance represents a record of a user's input and the corresponding
sentiment analysis outcome.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

# Create a base class for out ORM models using SQLAlchemy's declarative system
base = declarative_base()

class Sentiment(base):
    """
    SQLAlchemy model for sentiment records.

    Args:
        id (int): uniquely identifies the sentiment record.
        user_id (int): ID of the user that is associated with the sentiment input.
        user_text_input (str): The raw text from the user.
        sentiment_label (str): Sentiment classification label.
        confidence_score (float): Confidence score for the sentiment prediction.
        created_time (datetime): Timestamp of when the record was created.
    """
    
    __tablename__ = "sentiments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    user_text_input = Column(String, nullable=False)
    sentiment_label = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    created_time = Column(DateTime, default=datetime.now(timezone.utc))